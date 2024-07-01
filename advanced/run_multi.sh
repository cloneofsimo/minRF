export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=1

# put the above env var to ~/.deepspeed_env
# remove .deepseed_env if it exists
rm -f ~/.deepspeed_env
echo "NCCL_P2P_DISABLE=1" > ~/.deepspeed_env
echo "NCCL_P2P_LEVEL=NVL" >> ~/.deepspeed_env
echo "NCCL_SHM_DISABLE=1" >> ~/.deepspeed_env

# goes into ssh of each host in hostfile and run the following cmd
COMMAND="lsof /dev/nvidia* | awk '{print \$2}' | xargs -I {} kill {}"

# run the command on all hosts in hostfile
for host in `cat hostfiles`; do
    # host is in form of "xxx.xx.xxx.xx slots=8"
        host_ip=`echo $host | cut -d ' ' -f 1`
        echo "Running command on $host_ip"
        ssh -o StrictHostKeyChecking=no $host_ip $COMMAND
        # check if pytorch is installed
        ssh -o StrictHostKeyChecking=no $host_ip "python -c 'import torch; print(torch.randn(100).cuda())'"
        # remove /scratch/simo
        ssh -o StrictHostKeyChecking=no $host_ip "sudo rm -rf /scratch/simo && sudo mkdir -p /scratch/simo && sudo chmod 777 /scratch/simo && sudo chown -R nobody:nogroup /scratch/simo"
done

bash /home/simo/common_installations.sh


# --train_dir "/jfs/mds_relinked" \

deepspeed --hostfile=./hostfiles \
        main_t2i.py \
        --learning_rate 0.005 \
        --hidden_dim 3072 \
        --n_layers 36 \
        --n_heads 12 \
        --run_name 6.5b-dithybrid-36-24-node4-run-stage8 \
        --save_dir "/jfs/stage1_0621_stage8" \
        --num_train_epochs 200 \
        --train_batch_size 1024 \
        --per_device_train_batch_size 32 \
        --train_dir "/jfs/datacomp-1b-0-10k/1" \
        --seed 5 \
        --note "continue training" \
        --init_ckpt_path "/jfs/stage1_0621_stage7/model_57354/ema1.pt"