export WORLD_SIZE=8 #$(nvidia-smi -L | wc -l)
# deepspeed --num_gpus $WORLD_SIZE main.py --learning_rate 1e-3 --save_dir "/home/host/simo/ckpts/{}"
# lrs=(1e-4 2e-4 4e-4 8e-4)
# widths=(64 128 256)
loglr=(-8 -7 -6 -5 -4 -3)
widths=(32)

for width in "${widths[@]}"; do
    for loglr in "${loglr[@]}"; do
        lr=$(python -c "print(2**$loglr)")
        run_name="mup_lr_${lr}_width_${width}"
        echo "Running $run_name"
        deepspeed --num_gpus $WORLD_SIZE \
        main.py \
        --learning_rate $lr \
        --hidden_dim $width \
        --run_name $run_name \
        --save_dir "/home/host/simo/ckpts/${run_name}"
    done
done