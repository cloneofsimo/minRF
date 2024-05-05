export WORLD_SIZE=8 #$(nvidia-smi -L | wc -l)
# deepspeed --num_gpus $WORLD_SIZE main.py --learning_rate 1e-3 --save_dir "/home/host/simo/ckpts/{}"
lrs=(1e-3 3e-3 1e-2 3e-4 1e-4)
widths=(64 128 256)

for lr in "${lrs[@]}"; do
    for width in "${widths[@]}"; do
        run_name="lr_${lr}_width_${width}"
        deepspeed --num_gpus $WORLD_SIZE \
        main.py \
        --learning_rate $lr \
        --hidden_dim $width \
        --run_name $run_name \
        --save_dir "/home/host/simo/ckpts/${run_name}"
    done
done