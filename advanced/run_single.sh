export WORLD_SIZE=8

deepspeed --num_gpus $WORLD_SIZE \
        main.py \
        --learning_rate 0.01 \
        --hidden_dim 2560 \
        --n_layers 28 \
        --run_name largerun \
        --save_dir "/home/ubuntu/ckpts" \
        --num_train_epochs 200 \
        --train_batch_size 256 \
        --per_device_train_batch_size 16 \
