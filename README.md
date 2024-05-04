# Minimal Implementation of Rectified Flow

This repository contains a minimal implementation of the rectified flow models. I've taken [SD3](https://arxiv.org/abs/2403.03206) approach along with [LLaMA-DiT](https://github.com/Alpha-VLLM/LLaMA2-Accessory) architecture. [Unlike my previous repo](https://github.com/cloneofsimo/minDiffusion) this time I've decided to split the file into 2: The model implementation and actual code, but you don't have to look twice.

Everything is still self-contained, minimal, and easy to understand.

# 1. *Simple* Rectified Flow, for beginners

Install torch, pil, torchvision

```
pip install torch torchvision pillow
```

Run

```bash
python rf.py
```

to train the model on MNIST from scratch.

# 2. *Massive* Rectified Flow

This is for ambitious people who wants to train Imagenet instead. Don't worry! IMO Imagenet is the new MNIST, and we will use my [imagenet.int8](https://huggingface.co/datasets/cloneofsimo/imagenet.int8) dataset for this.

First go to advanced dir, download the dataset.

```bash
cd advanced
bash download.sh
```

Then run

```bash
bash run.sh
```

to train the model. This will train Imagenet from scratch.

# 3. Scalable Rectified Flow for absolute madlads

Check out this link for this. 