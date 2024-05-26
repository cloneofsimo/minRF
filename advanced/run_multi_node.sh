export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=1

# put the above env var to ~/.deepspeed_env
echo "NCCL_P2P_DISABLE=1" > ~/.deepspeed_env
echo "NCCL_P2P_LEVEL=NVL" >> ~/.deepspeed_env
echo "NCCL_SHM_DISABLE=1" >> ~/.deepspeed_env


deepspeed --hostfile=./hostfiles \
        main_t2i.py \
        --learning_rate 0.015 \
        --hidden_dim 2560 \
        --n_layers 28 \
        --run_name node-2-6.5b-run \
        --save_dir "/home/ubuntu/ckpts" \
        --num_train_epochs 200 \
        --train_batch_size 256 \
        --per_device_train_batch_size 8 \
        --train_dir "/jfs/datacomp-1b-0-10k/0/" \