export WORLD_SIZE=8 #$(nvidia-smi -L | wc -l)
deepspeed --num_gpus $WORLD_SIZE main.py