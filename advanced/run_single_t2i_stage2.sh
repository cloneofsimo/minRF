export WORLD_SIZE=8

deepspeed --num_gpus $WORLD_SIZE \
        main_t2i.py \
        --learning_rate 0.001 \
        --hidden_dim 2560 \
        --n_layers 24 \
        --run_name stage2 \
        --save_dir "/home/host/simo/ckpts/5b_highres" \
        --num_train_epochs 200 \
        --train_batch_size 16 \
        --per_device_train_batch_size 2 \
        --note "continue with smaller lr." \
        --train_dir "/home/host/simo/laionmds" \
	--seed 40
