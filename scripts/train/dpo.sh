#!/usr/bin/env bash
set -euo pipefail

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO
export PYTHONPATH=/datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 4 \
  --main_process_port 29538 \
  /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src/drpo/methods/dpo/trainer.py \
  --train_mode online \
  --pretrained_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/sd-turbo \
  --pickscore_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --pickscore_processor_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --pairs_jsonl /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/pairs.jsonl \
  --choice_model pickscore \
  --output_dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/outputs/dpo/online/latent_default \
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
  --eval_prompt_file /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/prompts/pickapicv2_test_unique.txt \
  --num_eval_prompts 10 \
  --eval_every_steps 0 \
  --seed 42 \
  --eval_seed 1234
