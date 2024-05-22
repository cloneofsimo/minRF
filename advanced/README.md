# LavenderFlow 5.6B v0

<p align="center">
    <img src="https://pbs.twimg.com/media/GOKPG_Ra0AELCOg?format=jpg&name=4096x4096" alt="large" width="600">
</p>

✅Latent, MMDiT, muP, CFM, FSDP, recaped, 768x768, T5

✅No strings attached, completely-open-every-step-of-the-way

✅Not SoTA😅(but.. not bad considering it was trained by one grad-student under total 3 weeks of development.) Severely undertrained!


# How to use

Once the model is trained with more compute, I will probably put more efforts to make this more accesible, but for now, have a look at [this notebook](https://github.com/cloneofsimo/minRF/blob/main/advanced/inference_t2i_example.ipynb)
You would need to download the latest checkpoint via

```bash
wget https://huggingface.co/cloneofsimo/lavenderflow-5.6B/resolve/highres-model_49153/ema1.pt
```

and load the checkpoint `ema1.pt` there.

# No paper, no blog, not raising cash, who am I and why did I do this?

I just do things for fun, no reasons but...

Foundation models seems to only belong in companies. I tell you, while it is tempting to say:
"You can't do this, you need team of engineers and researchers to make billion-scale foundation models"
it is not the case, and one bored grad student can obviously very easily pull this off. Now you know, you know.

# What did you do and how long did it take?

1. My first steps were to implement everything in torch and see if it works in MNIST, CIFAR-10. This took about 8 hours, including reading papers, implementing DiT, etc, everything.
2. Next was to scale up. I used muTransfer (basically i setup muP) to find optimal learning rate, and to do this I scaled upto 1B model with [imagenet.int8 dataset](https://github.com/cloneofsimo/imagenet.int8). Seeing the basin alignment, I scaled upto 5.6B parameters.
3. Then I moved to T2I training. I collected [Capfusion Dataset](https://huggingface.co/datasets/BAAI/CapsFusion-120M), which is subset of LAION-coco
4. I deduplicated with [SSCD embeddings](https://arxiv.org/abs/2202.10261) and [FAISS](https://faiss.ai/), using clustering method described in SD3.
5. I cached T5 (new [T5-large](https://huggingface.co/EleutherAI/pile-t5-large) from aieluther) and SDXL-VAE embeddings for all cropped 256x256 images.
6. For efficient sharding and to make better use of NFS, I put everything in [MosaicStreamingDataset](https://github.com/mosaicml/streaming) format, which was easy to use.
7. I rewrote my training code in DeepSpeed, utilizing Zero algorithm of stage-2, which does not shard the weights due to its small size. I got MFU of about 45 ~ 50 ~ 60% depending on the stage.
8. GPU GO BRR. there were some misconfigurations on the way, so I had to restart 3 times. This took about 550k total steps (3 days total)
9. Finally, I went through the same process all over again for [ye-pop dataset](https://huggingface.co/datasets/Ejafa/ye-pop), which is cleaned and recaptioned version of laion-pop dataset. This time with 768x768 resolution.

# Acknowlegement

Thank you to all authors of HDiT for bunch of pro tips and encouragements, specially @birchlabs.
Thanks to 서승현, who is my coworker at @naver who also shares pro tips, and studies deepspeed with me.
Thanks to @imbue.ai who provided most of the compute for this research project! Thanks to them, this is could be done at completely free cost!
Thanks to people at @fal.ai who provided compute and tips, specially @isidentital and @burkaygur !!

# What's the future?

@fal.ai has decided to provide more compute for this project in the future, so we decided to collaborate to make something much bigger, much better. You will see the upcoming version of this project with different names.
In the near future, I also hope to work on other modalities, but I have high chance of doing that at Naver without much being open sourced.
