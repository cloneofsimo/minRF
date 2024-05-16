loglr=(-7)
widths=(128)
gpuidx=(0)
masterports=(11201)
for width in "${widths[@]}"; do
    for idx in "${gpuidx[@]}"; do
        loglr_idx=$((idx))
        loglrv=${loglr[$loglr_idx]}
        masterport=${masterports[$idx]}
        lr=$(python -c "print(2**$loglrv)")
        run_name="layer48_mup_lr_${lr}_width_${width}"
        echo "Running $run_name"
        deepspeed --master_port $masterport --include=localhost:$idx \
        main.py \
        --learning_rate $lr \
        --hidden_dim $width \
        --run_name $run_name \
        --save_dir "/home/host/simo/ckpts/${run_name}" \
        --num_train_epochs 2 \
        --n_layers 48 \
        --train_batch_size 128 \
        --per_device_train_batch_size 128 &
    done
        % ${#loglr[@]}
        wait
done