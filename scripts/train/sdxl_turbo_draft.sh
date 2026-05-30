#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 4 \
  --main_process_port 29640 \
  "$PROJECT_ROOT"/src/drpo/methods/sdxl_draft/trainer.py \
  --pretrained_model_name_or_path "$PROJECT_ROOT"/models/stable-diffusion-xl-turbo \
  --prompt_file "$PROJECT_ROOT"/data/pickscore/train.txt \
  --output_dir "$PROJECT_ROOT"/outputs/sdxl-turbo-lora/draft/pickscore/default \
  --choice_model pickscore \
  --pickscore_model_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --pickscore_processor_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --resolution 512 \
  --batchsize_gen 24 \
  --train_batch_size 1 \
  --max_train_steps 5000 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --checkpointing_steps 100 \
  --dataloader_num_workers 2 \
  --report_to tensorboard \
  --ref_model_l2_weight 0.02 \
  --pickscore_loss_weight 1.0 \
  --score_std_weight 0.0 \
  --generation_chunk_size 8 \
  --vae_decode_chunk_size 1 \
  --use_lora
