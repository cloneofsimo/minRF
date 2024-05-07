export WORLD_SIZE=8 #$(nvidia-smi -L | wc -l)
# deepspeed --num_gpus $WORLD_SIZE main.py --learning_rate 1e-3 --save_dir "/home/host/simo/ckpts/{}"
# lrs=(1e-4 2e-4 4e-4 8e-4)
# widths=(64 128 256)
loglr=(-9 -8 -7 -6 -5 -4 -3 -2)
widths=(32 64)
gpuidx=(0 1 2 3 4 5 6 7)
masterports=(29500 29501 29502 29503 29504 29505 29506 29507)
for width in "${widths[@]}"; do
    for idx in "${gpuidx[@]}"; do
        loglr_idx=$((idx))
        loglr=${loglr[$loglr_idx]}
        masterport=${masterports[$idx]}
        lr=$(python -c "print(2**$loglr)")
        run_name="v2_mup_lr_${lr}_width_${width}"
        echo "Running $run_name"
        deepspeed --master_port $masterport --include=localhost:$idx \
        main.py \
        --learning_rate $lr \
        --hidden_dim $width \
        --run_name $run_name \
        --save_dir "/home/host/simo/ckpts/${run_name}" &
    done
        % ${#loglr[@]}
        wait
done