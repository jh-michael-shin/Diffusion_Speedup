In case the following code OOMs:

```py
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
)
transformer = FluxTransformer2DModel.from_pretrained(
    ckpt_id, 
    subfolder="transformer", 
    quantization_config=nf4_config,
    torch_dtype=bnb_4bit_compute_dtype,
)
```

Consider directly loading a quantized checkpoint such as: https://huggingface.co/sayakpaul/flux.1-dev-nf4-pkg. You can
specify `--quantized_model_path` to either supply a local path or a valid repo id on the Hub. 

Since we're fine-tuning a quantized checkpoint, it might need additional hyperparameter-tuning. The results shown
above are not optimal but it at least captures the subject well. 

Also, note that the training script currently DOES NOT support resuming from an intermediate checkpoint. 