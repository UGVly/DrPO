#!/usr/bin/env bash
set -euo pipefail

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO
export PYTHONPATH=/datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 8 \
  --main_process_port 29642 \
  /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src/drpo/methods/sdxl_drpo/trainer.py \
  --pretrained_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/stable-diffusion-xl-turbo \
  --prompt_file /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/prompts/pickapicv2_test_unique.txt \
  --output_dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/outputs/sdxl-turbo-lora/drpo/teacher_unet/lr1e-5_bs24_pos12_neg12_ga4_steps5000 \
  --feature_extractor teacher_unet \
  --teacher_feature_layers down_blocks.2,mid_block,up_blocks.0 \
  --teacher_feature_noise 0.1 \
  --teacher_feature_timestep 100 \
  --teacher_feature_pool_size 4 \
  --choice_model pickscore \
  --pickscore_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --pickscore_processor_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --resolution 512 \
  --batchsize_gen 24 \
  --num_pos_images 12 \
  --num_neg_images 12 \
  --max_train_steps 5000 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --checkpointing_steps 100 \
  --drifting_pos_weight 3000.0 \
  --drifting_neg_weight 3000.0 \
  --drifting_ref_weight 3000.0 \
  --drifting_ref_neg_weight 3000.0 \
  --drifting_ref_loss_weight 0.2 \
  --ref_model_l2_weight 0.02 \
  --use_lora
