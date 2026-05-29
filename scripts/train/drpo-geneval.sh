#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 4 \
  --main_process_port 29531 \
  "$PROJECT_ROOT"/src/drpo/methods/drpo/trainer.py \
  --train_mode online \
  --pretrained_model_name_or_path "$PROJECT_ROOT"/models/sd-turbo \
  --pairs_jsonl "$PROJECT_ROOT"/data/pairs.jsonl \
  --prompt_file "$PROJECT_ROOT"/third_party/geneval/prompts/evaluation_metadata.jsonl \
  --eval_prompt_file "$PROJECT_ROOT"/third_party/geneval/prompts/evaluation_metadata.jsonl \
  --choice_model geneval \
  --choice_score_normalize zscore \
  --geneval_repo "$PROJECT_ROOT"/third_party/geneval \
  --geneval_detector_path "$PROJECT_ROOT"/models/geneval_detector \
  --geneval_max_rollout_rounds 4 \
  --output_dir "$PROJECT_ROOT"/outputs/drpo/online/geneval_default \
  --drifting_mae_path "$PROJECT_ROOT"/drifting/mae_latent_256_torch.pth \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --batchsize_gen 24 \
  --num_pos_images 8 \
  --num_neg_images 8 \
  --max_train_steps 1000 \
  --learning_rate 1e-4 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 0 \
  --mixed_precision fp16 \
  --checkpointing_steps 100 \
  --eval_every_steps 0 \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --lora_target_modules to_q,to_k,to_v,to_out.0 \
  --drifting_feature_extractor mae \
  --drifting_feature_mode multi \
  --drifting_feature_keys layer4_mean,layer4_std,layer4_mean_2,layer4_std_2,layer4_mean_4,layer4_std_4 \
  --drifting_feature_aggregation sum \
  --drifting_kernel laplacian \
  --drifting_pref_r_list 0.02,0.05,0.2 \
  --drifting_ref_r_list 0.02,0.05,0.2 \
  --drifting_pos_weight 3000.0 \
  --drifting_neg_weight 3000.0 \
  --drifting_ref_weight 3000.0 \
  --drifting_ref_neg_weight 3000.0 \
  --drifting_ref_loss_weight 0.2 \
  --online_feature_top_fraction 1.0 \
  --use_lora
