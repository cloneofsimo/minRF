export WORLD_SIZE=8

deepspeed --num_gpus $WORLD_SIZE \
        main_t2i.py \
        --learning_rate 0.003 \
        --hidden_dim 2560 \
        --n_layers 24 \
        --run_name largerun_freeze_pd \
        --save_dir "/home/host/simo/ckpts/5b_fpd" \
        --num_train_epochs 200 \
        --train_batch_size 128 \
        --per_device_train_batch_size 16 \
        --note "I near-froze pixel dependency modules, init cond as 0"
