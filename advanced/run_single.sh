export WORLD_SIZE=8
lr=0.02

deepspeed --num_gpus $WORLD_SIZE \
        main.py \
        --learning_rate $lr \
        --hidden_dim 1024 \
        --run_name largerun \
        --save_dir "/home/host/simo/ckpts/largerun" \
        --num_train_epochs 2000

