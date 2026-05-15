#!/usr/bin/env bash
set -euo pipefail

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO
export PYTHONPATH=/datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 4 \
  --main_process_port 29671 \
  /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src/drpo/methods/vggflow/trainer.py \
  --pretrained_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/sd-turbo \
  --pickscore_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --pickscore_processor_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --aesthetic_clip_model_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/CLIP-ViT-L-14 \
  --aesthetic_ckpt_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/Aesthetic/sac+logos+ava1-l14-linearMSE.pth \
  --pairs_jsonl /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/pairs.jsonl \
  --choice_model pickscore \
  --output_dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/outputs/vggflow/pickscore/default \
  --mixed_precision fp16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --batchsize_gen 8 \
  --generation_timestep 999 \
  --generation_target_timestep 0 \
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
  --reward_scale 1000.0 \
  --eta_mode constant \
  --rgrad_clip_threshold 1.0 \
  --rgrad_quantile 0.8 \
  --rgrad_jitter_count 1 \
  --rgrad_jitter_std 0.0 \
  --reward_mask_threshold 0.0 \
  --unet_reg_scale 0.0 \
  --vae_decode_chunk_size 4 \
  --eval_prompt_file /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/prompts/pickapicv2_test_unique.txt \
  --num_eval_prompts 10 \
  --eval_every_steps 0 \
  --seed 42 \
  --eval_seed 1234
