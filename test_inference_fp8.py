import os
PATH = '/workspace/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

from diffusers import FluxPipeline
from torchao.quantization import autoquant
import torch 

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")


# If you are using "autoquant" then you should compile first and then
# apply autoquant.
# pipeline.transformer.to(memory_format=torch.channels_last)
# pipeline.transformer = torch.compile(
#   pipeline.transformer, mode="reduce-overhead", fullgraph=True
# )

# pipeline.transformer = autoquant(pipeline.transformer, error_on_unseen=False)
image = pipeline(
    "a dog surfing on moon", guidance_scale=3.5, num_inference_steps=50
).images[0]



"""
bfloat16 with autoquant with torch.compile(mode = max-autotune)
2024-12-27T12:19:05Z WARN: very high memory utilization: 116.4GiB / 116.4GiB (100 %)
2024-12-27T12:19:08Z WARN: container is unhealthy: triggered memory limits (OOM)
2024-12-27T12:19:24Z WARN: container is unhealthy: triggered memory limits (OOM)
2024-12-27T12:19:29Z WARN: container is unhealthy: triggered memory limits (OOM)

with autoquant with torch.compile(mode = reduce-overhead)
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 90.00 MiB. GPU 0 has a total capacity of 23.54 GiB of which 63.75 MiB is free. 
Process 84663 has 23.47 GiB memory in use. Of the allocated memory 23.09 GiB is allocated by PyTorch, 
and 9.65 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try 
setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  
See documentation for Memory Management  
(https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

Need to quantize to run in single 4090 GPU
"""