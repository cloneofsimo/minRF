export WORLD_SIZE=8

deepspeed --num_gpus $WORLD_SIZE \
        main_t2i.py \
        --learning_rate 0.002 \
        --hidden_dim 2560 \
        --n_layers 24 \
        --run_name largerun_freeze_pd_2 \
        --save_dir "/home/host/simo/ckpts/5b_cont_2" \
        --num_train_epochs 200 \
        --train_batch_size 128 \
        --per_device_train_batch_size 16 \
        --note "continue with smaller lr." \
	--seed 40
