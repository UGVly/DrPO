#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 4 \
  --main_process_port 29642 \
  "$PROJECT_ROOT"/src/drpo/methods/sdxl_drpo/trainer.py \
  --pretrained_model_name_or_path "$PROJECT_ROOT"/models/stable-diffusion-xl-turbo \
  --prompt_file "$PROJECT_ROOT"/data/prompts/pickapicv2_test_unique.txt \
  --output_dir "$PROJECT_ROOT"/outputs/sdxl-turbo-lora/drpo/teacher_unet/default \
  --feature_extractor teacher_unet \
  --teacher_feature_layers down_blocks.2,mid_block,up_blocks.0 \
  --teacher_feature_noise 0.1 \
  --teacher_feature_timestep 100 \
  --teacher_feature_pool_size 4 \
  --choice_model pickscore \
  --pickscore_model_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --pickscore_processor_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --resolution 512 \
  --batchsize_gen 24 \
  --num_pos_images 12 \
  --num_neg_images 12 \
  --max_train_steps 5000 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --lr_warmup_steps 0 \
  --mixed_precision bf16 \
  --checkpointing_steps 100 \
  --drifting_pos_weight 3000.0 \
  --drifting_neg_weight 2000.0 \
  --drifting_ref_weight 3500.0 \
  --drifting_ref_neg_weight 5000.0 \
  --drifting_ref_loss_weight 0.25 \
  --ref_model_l2_weight 0.03 \
  --feature_diversity_weight 1.0 \
  --feature_diversity_margin_scale 0.80 \
  --use_lora
