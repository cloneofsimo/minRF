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
from deepspeed import get_accelerator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils import logger
from streaming import StreamingDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

import wandb

from ..dit import DiT_Llama


class RF(torch.nn.Module):
    def __init__(self, model, ln=True):
        super().__init__()
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
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
        return (x / 255.0 - 0.5) * 24.0


_encodings["uint8"] = uint8


@click.command()
@click.option("--local_rank", default=-1, help="Local rank")
@click.option("--num_train_epochs", default=2000, help="Number of training epochs")
@click.option("--learning_rate", default=1e-4, help="Learning rate")
@click.option("--offload", default=False, help="Offload")
@click.option("--train_batch_size", default=256, help="Train batch size")
@click.option(
    "--per_device_train_batch_size", default=32, help="Per device train batch size"
)
@click.option("--zero_stage", default=2, help="Zero stage")
@click.option("--seed", default=42, help="Seed")
@click.option("--run_name", default=None, help="Run name")
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
    train_dir="../vae_mds",
    skipped_ema_step=16,
):
    # first, set the seed
    set_seed(seed)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

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
    os.environ["LOCAL_WORLD_SIZE"] = str(os.environ.get("WORLD_SIZE"))

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
        "gradient_clipping": 1.0,
    }

    torch.distributed.barrier()

    global_rank = torch.distributed.get_rank()

    ##### DEFINE model, dataset, sampler, dataloader, optim, schedular

    with deepspeed.zero.Init(enabled=(zero_stage == 3)):

        ddpm = RF(
            DiT_Llama(
                in_channels=4,
                dim=512,
                n_layers=12,
                n_heads=16,
                num_classes=1000,
                ffn_dim_multiplier=4,
            ),
            1000,
        ).cuda()

    ema_state_dict1 = extract_model_state_dict_deepspeed(ddpm, global_rank)
    ema_state_dict2 = extract_model_state_dict_deepspeed(ddpm, global_rank)

    total_params = sum(p.numel() for p in ddpm.parameters())
    size_in_bytes = total_params * 4
    size_in_gb = size_in_bytes / (1024**3)
    logger.info(
        f"Model Size: {size_in_bytes}, {size_in_gb} GB, Total Param Count: {total_params / 1e6} M"
    )

    util.clean_stale_shared_memory()
    # barrier
    torch.distributed.barrier()

    train_dataset = StreamingDataset(
        local=train_dir,
        remote=None,
        split=None,
        shuffle=True,
        shuffle_algo="naive",
        num_canonical_nodes=1,
        batch_size=per_device_train_batch_size,
    )

    print(f"\n\n######-----Dataset loaded: {len(train_dataset)}")
    print(
        f"Rank: {os.environ.get('RANK')}, Local Rank: {os.environ.get('LOCAL_WORLD_SIZE')}, Global Rank: {global_rank}"
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
    )

    torch.distributed.barrier()

    optimizer = torch.optim.AdamW(
        ddpm.eps_model.parameters(), lr=learning_rate, weight_decay=0.1
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_train_epochs * math.ceil(len(dataloader)),
    )

    ddpm.train()

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=ddpm, config=ds_config, lr_scheduler=lr_scheduler, optimizer=optimizer
    )

    global_step = 0

    if global_rank == 0:
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
    ##### actual training loop

    if global_rank == 0:
        wandb.init(
            project="rf_in1k",
            name=run_name,
            config={
                "num_train_epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "offload": offload,
                "train_batch_size": train_batch_size,
                "per_device_train_batch_size": per_device_train_batch_size,
                "zero_stage": zero_stage,
                "seed": seed,
            },
        )

    for i in range(num_train_epochs):
        pbar = tqdm(dataloader)

        for batch in pbar:

            x = (
                batch["vae_output"].reshape(-1, 4, 32, 32).to(device).to(torch.bfloat16)
                * 0.13025
            )
            t = torch.randint(1, ddpm.n_T + 1, (x.shape[0],)).to(x.device)
            y = torch.tensor(list(map(int, batch["label"]))).long().to(x.device)

            loss, info = model_engine(x, t, y)
            model_engine.backward(loss)
            model_engine.step()

            get_accelerator().empty_cache()
            norm = model_engine.get_global_grad_norm()

            global_step += 1
            if global_rank == 0:
                batchwise_loss = info["batchwise_loss"]
                # check t-wise loss
                for t, l in batchwise_loss:
                    lossbin[int(t * 10)] += l
                    losscnt[int(t * 10)] += 1

            if global_step % skipped_ema_step == 1:

                current_state_dict = extract_model_state_dict_deepspeed(
                    ddpm, global_rank
                )

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

            pbar.set_description(
                f"norm: {norm}, loss: {loss.item()}, global_step: {global_step}"
            )

            if global_step % 800 == 1:

                os.makedirs(f"./ckpt/model_{global_step}", exist_ok=True)
                save_zero_three_model(
                    model_engine,
                    global_rank,
                    f"./ckpt/model_{global_step}/",
                    zero_stage=zero_stage,
                )

                # save ema weights
                if global_rank == 0:
                    torch.save(ema_state_dict1, f"./ckpt/model_{global_step}/ema1.pt")
                    torch.save(ema_state_dict2, f"./ckpt/model_{global_step}/ema2.pt")

                print(f"Model saved at {global_step}")


if __name__ == "__main__":
    main()
