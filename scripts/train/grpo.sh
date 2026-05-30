#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 8 \
  --main_process_port 29670 \
  "$PROJECT_ROOT"/baselines/grpo/train_lora.py \
  --pretrained_model_name_or_path "$PROJECT_ROOT"/models/sd-turbo \
  --pickscore_model_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --pickscore_processor_name_or_path "$PROJECT_ROOT"/models/PickScore_v1 \
  --train_prompt_file "$PROJECT_ROOT"/data/pickscore/train.txt \
  --choice_model pickscore \
  --output_dir "$PROJECT_ROOT"/outputs/grpo/pickscore/lr1e-5_bs24_ga4_steps5000 \
  --mixed_precision bf16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --batchsize_gen 24 \
  --generation_timestep 999 \
  --generation_target_timestep 0 \
  --max_train_steps 5000 \
  --learning_rate 1e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 0 \
  --checkpointing_steps 100 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --lora_target_modules to_q,to_k,to_v,to_out.0 \
  --neighbor_sigma 0.3 \
  --neighbor_num_anchors 4 \
  --neighbor_p_norm 0.8 \
  --neighbor_distance_temperature 1.0 \
  --neighbor_distance_reduction mean \
  --ppo_clip_range 0.2 \
  --max_log_ratio 10.0 \
  --advantage_clip 5.0 \
  --advantage_scale 1.0 \
  --policy_kl_weight 0.0 \
  --ref_model_l2_weight 0.02 \
  --vae_decode_chunk_size 4 \
  --reward_score_batch_size 128 \
  --reward_cache_interval 1 \
  --eval_prompt_file "$PROJECT_ROOT"/data/pickscore/test.txt \
  --num_eval_prompts 10 \
  --eval_every_steps 0 \
  --seed 42 \
  --eval_seed 1234
