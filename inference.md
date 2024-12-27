When loading the LoRA params (that were obtained on a quantized base model) and merging them into the base model, it is recommended to first dequantize the base model, merge the LoRA params into it, and then quantize the model again. This is because merging into 4bit quantized models can lead to some rounding errors. Below, we provide an end-to-end example:

1. First, load the original model and merge the LoRA params into it:

```py
from diffusers import FluxPipeline 
import torch 

ckpt_id = "black-forest-labs/FLUX.1-dev"
pipeline = FluxPipeline.from_pretrained(
    ckpt_id, text_encoder=None, text_encoder_2=None, torch_dtype=torch.float16
)
pipeline.load_lora_weights("yarn_art_lora_flux_nf4", weight_name="pytorch_lora_weights.safetensors")
pipeline.fuse_lora()
pipeline.unload_lora_weights()

pipeline.transformer.save_pretrained("fused_transformer")
```

2. Quantize the model and run inference

```py
from diffusers import AutoPipelineForText2Image, FluxTransformer2DModel, BitsAndBytesConfig
import torch

ckpt_id = "black-forest-labs/FLUX.1-dev"
bnb_4bit_compute_dtype = torch.float16
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
)
transformer = FluxTransformer2DModel.from_pretrained(
    "fused_transformer",
    quantization_config=nf4_config,
    torch_dtype=bnb_4bit_compute_dtype,
)
pipeline = AutoPipelineForText2Image.from_pretrained(
    ckpt_id, transformer=transformer, torch_dtype=bnb_4bit_compute_dtype
)
pipeline.enable_model_cpu_offload()

image = pipeline(
    "a puppy in a pond, yarn art style", num_inference_steps=28, guidance_scale=3.5, height=768
).images[0]
image.save("yarn_merged.png")
```

|   Dequantize, merge, quantize   |   Merging directly into quantized model   |
|-------|-------|
| ![Image A](https://gist.github.com/user-attachments/assets/4b0e9059-4777-4eaa-900c-09d3435f55d0) | ![Image B](https://gist.github.com/user-attachments/assets/e56e8490-b2e8-4b94-9199-dd2ca42aba34) |

As we can notice the first column result follows the style more closely. 