#!/usr/bin/env bash
set -euo pipefail

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO
export PYTHONPATH=/datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 8 \
  --main_process_port 29671 \
  /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src/drpo/methods/sdxl_grpo/trainer.py \
  --pretrained_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/stable-diffusion-xl-turbo \
  --prompt_file /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/prompts/pickapicv2_test_unique.txt \
  --pickscore_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --pickscore_processor_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --choice_model pickscore \
  --choice_score_normalize zscore \
  --output_dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/outputs/sdxl-turbo-lora/grpo/pickscore/lr1e-5_bs24_ga4_steps5000 \
  --model_variant fp16 \
  --mixed_precision bf16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --batchsize_gen 24 \
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
  --vae_decode_chunk_size 1 \
  --seed 42
