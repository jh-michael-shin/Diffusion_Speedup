Accelerate config:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: NO
downcast_bf16: 'no'
enable_cpu_affinity: true
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Training command:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file=accelerate.yaml train_dreambooth_lora_flux_nf4.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset_name="Norod78/Yarn-art-style" \
  --instance_prompt="a puppy, yarn art style" \
  --output_dir="yarn_art_lora_flux_nf4" \
  --mixed_precision="bf16" \
  --use_8bit_adam \
  --weighting_scheme="none" \
  --resolution=1024 \
  --train_batch_size=1 \
  --repeats=1 \
  --learning_rate=1e-4 \
  --guidance_scale=1 \
  --report_to="wandb" \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --guidance_scale=1 \
  --cache_latents \
  --rank=4 \
  --max_train_steps=700 \
  --seed="0" \
  --push_to_hub
  ```