#!/usr/bin/env bash
set -euo pipefail

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO
export PYTHONPATH=/datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 8 \
  --main_process_port 29640 \
  /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src/drpo/methods/sdxl_draft/trainer.py \
  --pretrained_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/stable-diffusion-xl-turbo \
  --prompt_file /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/prompts/pickapicv2_test_unique.txt \
  --output_dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/outputs/sdxl-turbo-lora/draft/pickscore/default \
  --choice_model pickscore \
  --pickscore_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --pickscore_processor_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --resolution 512 \
  --batchsize_gen 24 \
  --train_batch_size 1 \
  --max_train_steps 2000 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --checkpointing_steps 100 \
  --dataloader_num_workers 2 \
  --report_to tensorboard \
  --ref_model_l2_weight 0.02 \
  --pickscore_loss_weight 1.0 \
  --score_std_weight 0.0 \
  --vae_decode_chunk_size 1 \
  --use_lora
