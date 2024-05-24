deepspeed --hostfile=./hostfiles \
        main_t2i.py \
        --learning_rate 0.015 \
        --hidden_dim 32 \
        --n_layers 28 \
        --run_name largerun \
        --save_dir "/home/ubuntu/ckpts" \
        --num_train_epochs 200 \
        --train_batch_size 256 \
        --per_device_train_batch_size 16 \
        --train_dir "/jfs/datacomp-1b-0-10k/0/" \