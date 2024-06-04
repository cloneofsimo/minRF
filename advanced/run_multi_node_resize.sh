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
done



deepspeed --hostfile=./hostfiles \
        main_t2i_highres.py \
        --learning_rate 0.0366 \
        --hidden_dim 2560 \
        --n_layers 36 \
        --run_name node-2-highres \
        --save_dir "/home/ubuntu/ckpts_36L_2_highres" \
        --num_train_epochs 200 \
        --train_batch_size 384 \
        --per_device_train_batch_size 6 \
        --train_dir "/home/ubuntu/laionpop" \
        --seed 3 \
        --note "Laion Pop Aesthetic fine tuning" \
        --resize_pe_at_initialization True \
        --vaeres 128