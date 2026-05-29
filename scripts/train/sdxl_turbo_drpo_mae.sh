#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 8 \
  --main_process_port 29641 \
  "$PROJECT_ROOT"/src/drpo/methods/sdxl_drpo/trainer.py \
  --pretrained_model_name_or_path "$PROJECT_ROOT"/models/stable-diffusion-xl-turbo \
  --mae_model_name_or_path "$PROJECT_ROOT"/models/facebook-vit-mae-base \
  --prompt_file "$PROJECT_ROOT"/data/prompts/pickapicv2_test_unique.txt \
  --output_dir "$PROJECT_ROOT"/outputs/sdxl-turbo-lora/drpo/mae/default \
  --feature_extractor mae \
  --mae_feature_keys layer12_patch_mean,layer12_patch_std,layer12_cls \
  --choice_model pickscore \
  --pickscore_model_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --pickscore_processor_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --resolution 512 \
  --batchsize_gen 16 \
  --num_pos_images 8 \
  --num_neg_images 8 \
  --max_train_steps 5000 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --lr_warmup_steps 0 \
  --mixed_precision bf16 \
  --checkpointing_steps 100 \
  --drifting_pos_weight 3000.0 \
  --drifting_neg_weight 3000.0 \
  --drifting_ref_weight 3000.0 \
  --drifting_ref_neg_weight 3000.0 \
  --drifting_ref_loss_weight 0.2 \
  --ref_model_l2_weight 0.02 \
  --use_lora
