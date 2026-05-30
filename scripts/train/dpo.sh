#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 4 \
  --main_process_port 29538 \
  "$PROJECT_ROOT"/baselines/dpo/train_lora.py \
  --train_mode online \
  --pretrained_model_name_or_path "$PROJECT_ROOT"/models/sd-turbo \
  --pickscore_model_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --pickscore_processor_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --train_prompt_file "$PROJECT_ROOT"/data/pickscore/train.txt \
  --choice_model pickscore \
  --output_dir "$PROJECT_ROOT"/outputs/dpo/online/latent_default \
  --mixed_precision fp16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --batchsize_gen 24 \
  --max_train_steps 1000 \
  --learning_rate 1e-4 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 0 \
  --checkpointing_steps 100 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --lora_target_modules to_q,to_k,to_v,to_out.0 \
  --preference_space latent \
  --dpo_beta 30.0 \
  --dpo_loss_weight 1.0 \
  --ref_model_l2_weight 0.0 \
  --dpo_distance_metric l2 \
  --anchor_encode_mode mode \
  --num_pos_images 8 \
  --num_neg_images 8 \
  --eval_prompt_file "$PROJECT_ROOT"/data/pickscore/test.txt \
  --num_eval_prompts 10 \
  --eval_every_steps 0 \
  --seed 42 \
  --eval_seed 1234
