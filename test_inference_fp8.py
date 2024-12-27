from diffusers import FluxPipeline
from torchao.quantization import autoquant
import torch 

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")


# If you are using "autoquant" then you should compile first and then
# apply autoquant.
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
  pipeline.transformer, mode="max-autotune", fullgraph=True
)

pipeline.transformer = autoquant(pipeline.transformer, error_on_unseen=False)
image = pipeline(
    "a dog surfing on moon", guidance_scale=3.5, num_inference_steps=50
).images[0]
