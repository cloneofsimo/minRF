# Mostly Copy-paste from https://github.com/cloneofsimo/min-max-in-dit.

import math
import os
import random
from typing import Any

import click
import deepspeed
import numpy as np
import streaming.base.util as util
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from streaming import StreamingDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

import wandb
from mmdit import MMDiT


class RF(torch.nn.Module):
    def __init__(self, model, ln=True):
        super().__init__()
        self.model = model
        self.ln = ln
        self.stratified = False

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            if self.stratified:
                # stratified sampling of normals
                # first stratified sample from uniform
                quantiles = torch.linspace(0, 1, b + 1).to(x.device)
                z = quantiles[:-1] + torch.rand((b,)).to(x.device) / b
                # now transform to normal
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                t = torch.sigmoid(z)
            else:
                nt = torch.randn((b,)).to(x.device)
                t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        # make t, zt into same dtype as x
        zt, t = zt.to(x.dtype), t.to(x.dtype)
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), {"batchwise_loss": ttloss}

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

    @torch.no_grad()
    def sample_with_xps(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)
            x = z - i * dt * vc
            z = z - dt * vc
            images.append(x)
        return images


def _z3_params_to_fetch(param_list):
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = zero_stage == 3
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model.module if hasattr(model, "module") else model
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, "ds_id"):
                with deepspeed.zero.GatheredParameters(
                    _z3_params_to_fetch([v]), enabled=zero_stage_3
                ):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


@torch.no_grad()
def extract_model_state_dict_deepspeed(model, global_rank):
    output_state_dict = {}
    for k, v in model.named_parameters():
        if hasattr(v, "ds_id"):
            with deepspeed.zero.GatheredParameters(
                _z3_params_to_fetch([v]), enabled=True
            ):
                v_p = v.data.cpu()
        else:
            v_p = v.cpu()

        if global_rank == 0:
            output_state_dict[k] = v_p.detach()

    return output_state_dict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 28.0


class np16(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float16)


_encodings["np16"] = np16
_encodings["uint8"] = uint8


@click.command()
@click.option("--local_rank", default=-1, help="Local rank")
@click.option("--num_train_epochs", default=2, help="Number of training epochs")
@click.option("--learning_rate", default=3e-3, help="Learning rate")
@click.option("--offload", default=False, help="Offload")
@click.option("--train_batch_size", default=256, help="Total Train batch size")
@click.option(
    "--per_device_train_batch_size", default=128, help="Per device train batch size"
)
@click.option("--zero_stage", default=2, help="Zero stage, from 0 to 3")
@click.option("--seed", default=42, help="Seed for rng")
@click.option("--run_name", default=None, help="Run name that will be used for wandb")
@click.option(
    "--train_dir",
    default="/home/host/simo/capfusion_mds",
    help="Train dir that MDS can read",
)
@click.option(
    "--skipped_ema_step",
    default=1024,
    help="Skipped EMA step. Karras EMA will save model every skipped_ema_step",
)
@click.option("--weight_decay", default=0.1, help="Weight decay")
@click.option(
    "--hidden_dim",
    default=256,
    help="Hidden dim, this will mainly determine the model size",
)
@click.option(
    "--n_layers",
    default=12,
    help="Number of layers, this will mainly determine the model size",
)
@click.option("--save_dir", default="./ckpt", help="Save dir for model")
@click.option(
    "--lr_frozen_factor",
    default=1.0,
    help="Learning rate for (nearly) frozen layers. You would want this less than 1.",
)
@click.option("--note", default="hi", help="Note for wandb")
@click.option("--vaeres", default=96, help="VAE resolution. 96 x 96 by default")
def main(
    local_rank,
    train_batch_size,
    per_device_train_batch_size,
    num_train_epochs,
    learning_rate,
    offload=False,
    zero_stage=2,
    seed=42,
    run_name=None,
    train_dir="./vae_mds",
    skipped_ema_step=16,
    weight_decay=0.1,
    hidden_dim=256,
    n_layers=12,
    save_dir="./ckpt",
    lr_frozen_factor=1.0,
    note="hi",
    vaeres=96,
):

    # first, set the seed
    set_seed(seed)

    if run_name is None:
        run_name = (
            f"LR:{learning_rate}__num_train_epochs:{num_train_epochs}_offload:{offload}"
        )

    if local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(local_rank)
        device = torch.device(get_accelerator().device_name(), local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    # set LOCAL_WORLD_SIZE to 8

    offload_device = "cpu" if offload else "none"

    ds_config = {
        "train_micro_batch_size_per_gpu": per_device_train_batch_size,
        "train_batch_size": train_batch_size,
        "zero_optimization": {
            "stage": zero_stage,
            "offload_param": {"device": offload_device},
            "offload_optimizer": {"device": offload_device},
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False,
        },
        "bfloat16": {"enabled": True},
        "gradient_clipping": 0.3,
    }

    torch.distributed.barrier()

    global_rank = torch.distributed.get_rank()

    ##### DEFINE model, dataset, sampler, dataloader, optim, schedular

    with deepspeed.zero.Init(enabled=(zero_stage == 3)):

        rf = RF(
            MMDiT(
                in_channels=4,
                out_channels=4,
                dim=hidden_dim,
                global_conddim=hidden_dim,
                n_layers=n_layers,
                n_heads=8,
                cond_seq_dim=1024,
            ),
            True,
        ).cuda()
        # rf.load_state_dict(
        #     torch.load(
        #         "rf_450k_ema1.pt",
        #         map_location="cpu",
        #     ),
        #     strict=False,
        # )

        #rf.model.extend_pe((32, 32), (vaeres, vaeres))

    ema_state_dict1 = extract_model_state_dict_deepspeed(rf, global_rank)
    ema_state_dict2 = extract_model_state_dict_deepspeed(rf, global_rank)

    total_params = sum(p.numel() for p in rf.parameters())
    size_in_bytes = total_params * 4
    size_in_gb = size_in_bytes / (1024**3)
    print(
        f"Model Size: {size_in_bytes}, {size_in_gb} GB, Total Param Count: {total_params / 1e6} M"
    )

    util.clean_stale_shared_memory()
    # barrier
    torch.distributed.barrier()

    os.environ['LOCAL_WORLD_SIZE'] = str(8)
# WORLD_SIZE: Total number of processes to launch across all nodes.
# LOCAL_WORLD_SIZE: Total number of processes to launch for each node.
# RANK: Rank of the current process, which is the range between 0 to WORLD_SIZE - 1.
# MASTER_ADDR: The hostname for the rank-zero process.
# MASTER_PORT: The port for the rank-zero process.

    for varname in [
        "RANK",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]:
        assert os.environ.get(varname) is not None, f"{varname} is not set"
        print(f"{varname}: {os.environ.get(varname)}")

    locdir = f'/tmp/mdstemp_0'
    # cleanup if rank0
    if local_rank == 0:
        os.system(f"rm -rf {locdir}")
        # make
        os.makedirs(locdir, exist_ok=True)

    torch.distributed.barrier()

    train_dataset = StreamingDataset(
        local=locdir,
        remote=train_dir,
        split=None,
        shuffle=True,
        shuffle_algo="py1e",
        num_canonical_nodes=2,
        batch_size=per_device_train_batch_size,
        shuffle_seed=seed,
        shuffle_block_size = 10 * 4000
    )


    right_pad = lambda x: torch.cat(
        [x, torch.zeros(128 - x.size(0), 1024).to(x.device)], dim=0
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        num_workers=8,
        collate_fn=lambda x: {
            "vae_output": torch.stack([torch.tensor(xx["vae_output"]) for xx in x]),
            "t5emb": torch.stack(
                [right_pad(torch.tensor(xx["t5emb"]).reshape(-1, 1024)) for xx in x]
            ),
        },
    )

    torch.distributed.barrier()
    ## Config muP-learning rate.
    no_decay_name_list = [
        "bias",
        "norm",
    ]

    input_tensors = [
        'cond_seq_linear', 'init_x_linear'
    ]

    small_train_name_list = ["w2q", "w2k", "w2v", "w2o", "mlpX", "modX"]

    optimizer_grouped_parameters = []
    final_optimizer_settings = {}

    for n, p in rf.named_parameters():
        group_parameters = {}
        if p.requires_grad:
            if any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["weight_decay"] = 0.0
            else:
                group_parameters["weight_decay"] = weight_decay

            # Define learning rate for specific types of params

            if "embed" in n or any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["lr"] = learning_rate * 0.1
            elif any(ipt in n for ipt in input_tensors):
                input_dim = p.shape[0]
                group_parameters["lr"] = learning_rate * (16 / input_dim)
            else:
                group_parameters["lr"] = learning_rate * (32 / hidden_dim)

            if any(ndnl in n for ndnl in small_train_name_list):
                group_parameters["lr"] = group_parameters["lr"] * lr_frozen_factor

            group_parameters["params"] = [p]
            final_optimizer_settings[n] = {
                "lr": group_parameters["lr"],
                "wd": group_parameters["weight_decay"],
                'shape': str(list(p.shape))
            }
            optimizer_grouped_parameters.append(group_parameters)

    if global_rank == 0:
        # mkdir and dump optimizer settings
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f"{save_dir}/optimizer_settings.txt", "w") as f:
            # format
            for k, v in sorted(final_optimizer_settings.items()):
                f.write(f"{k}: {v}\n")

        

    AdamOptimizer = torch.optim.AdamW

    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.999)
    )

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=300, num_training_steps=1e6
    )

    rf.train()

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rf, config=ds_config, lr_scheduler=lr_scheduler, optimizer=optimizer
    )

    global_step = 0

    if global_rank == 0:
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
    ##### actual training loop

    if global_rank == 0:
        wandb.init(
            project="rf_t2i_mup",
            name=run_name,
            config={
                "num_train_epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "offload": offload,
                "train_batch_size": train_batch_size,
                "per_device_train_batch_size": per_device_train_batch_size,
                "zero_stage": zero_stage,
                "seed": seed,
                "train_dir": train_dir,
                "skipped_ema_step": skipped_ema_step,
                "weight_decay": weight_decay,
                "hidden_dim": hidden_dim,
            },
            notes=note,
        )

    for i in range(num_train_epochs):
        pbar = tqdm(dataloader)

        for batch in pbar:

            x = (
                batch["vae_output"]
                .reshape(-1, 4, vaeres, vaeres)
                .to(device)
                .to(torch.bfloat16)
                * 0.13025
            )

            cond = batch["t5emb"].reshape(-1, 128, 1024).to(device).to(torch.bfloat16)
            
            loss, info = model_engine(x, {"c_seq": cond})
            model_engine.backward(loss)
            model_engine.step()

            norm = model_engine.get_global_grad_norm()

            global_step += 1
            if global_rank == 0:
                batchwise_loss = info["batchwise_loss"]
                # check t-wise loss
                for t, l in batchwise_loss:
                    lossbin[int(t * 10)] += l
                    losscnt[int(t * 10)] += 1

                if global_step % 16 == 0:
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "train_grad_norm": norm,
                            "value": value,
                            "ema1_of_value": ema1_of_value,
                            "ema2_of_value": ema2_of_value,
                            **{
                                f"lossbin_{i}": lossbin[i] / losscnt[i]
                                for i in range(10)
                            },
                        }
                    )
                    # reset
                    lossbin = {i: 0 for i in range(10)}
                    losscnt = {i: 1e-6 for i in range(10)}

            if global_step % skipped_ema_step == 1:

                current_state_dict = extract_model_state_dict_deepspeed(rf, global_rank)

                if global_rank == 0:

                    # update ema.
                    BETA1 = (1 - 1 / global_step) ** (1 + 16)
                    BETA2 = (1 - 1 / global_step) ** (1 + 9)

                    # adjust beta for skipped-ema
                    BETA1 = 1 - (1 - BETA1) * skipped_ema_step
                    BETA1 = max(0, BETA1)
                    BETA2 = 1 - (1 - BETA2) * skipped_ema_step
                    BETA2 = max(0, BETA2)

                    value = None
                    ema1_of_value = None
                    ema2_of_value = None

                    for k, v in current_state_dict.items():
                        ema_state_dict1[k] = (
                            BETA1 * ema_state_dict1[k] + (1 - BETA1) * v
                        )
                        ema_state_dict2[k] = (
                            BETA2 * ema_state_dict2[k] + (1 - BETA2) * v
                        )
                        # log 1st value for sanity check
                        if value is None:
                            value = v.half().flatten()[0].item()
                            ema1_of_value = (
                                ema_state_dict1[k].half().flatten()[0].item()
                            )
                            ema2_of_value = (
                                ema_state_dict2[k].half().flatten()[0].item()
                            )

            pbar.set_description(
                f"norm: {norm}, loss: {loss.item()}, global_step: {global_step}"
            )

            if global_step % 4096 == 1:
        
                os.makedirs(f"{save_dir}/model_{global_step}", exist_ok=True)
                save_zero_three_model(
                    model_engine,
                    global_rank,
                    f"{save_dir}/model_{global_step}/",
                    zero_stage=zero_stage,
                )

                # save ema weights
                if global_rank == 0:
                    torch.save(
                        ema_state_dict1, f"{save_dir}/model_{global_step}/ema1.pt"
                    )
                    torch.save(
                        ema_state_dict2, f"{save_dir}/model_{global_step}/ema2.pt"
                    )

                print(f"Model saved at {global_step}")


if __name__ == "__main__":
    main()
