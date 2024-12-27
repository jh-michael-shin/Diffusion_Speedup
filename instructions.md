Make sure to install the following libraries and ensure an NVIDIA GPU environment (with at least 24GB VRAM):

```bash
pip install -U bitsandbytes peft
pip install git+https://github.com/huggingface/diffusers
pip install git+https://github.com/huggingface/transformers
```

I didn't test the underlying scripts on a 16GB card but they might work. I used a 4090 card for testing. 

The training script `train_dreambooth_lora_flux_nf4.py` is hacked up version of the original `diffusers` training script:
https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py. 

> [!NOTE]  
> Training takes at least 1.5 hours on the settings I tried out.