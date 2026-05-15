#!/usr/bin/env bash
set -euo pipefail

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO
export PYTHONPATH=/datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 4 \
  --main_process_port 29530 \
  /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src/drpo/training/sdturbo_full.py \
  --train_mode online \
  --pretrained_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/sd-turbo \
  --pickscore_model_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --pickscore_processor_name_or_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/PickScore_v1 \
  --pairs_jsonl /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/pairs.jsonl \
  --prompt_file /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/prompts/pickapicv2_test_unique.txt \
  --eval_prompt_file /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/prompts/pickapicv2_test_unique.txt \
  --choice_model pickscore \
  --output_dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/outputs/drpo/online/full_default \
  --drifting_mae_path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/drifting/mae_latent_256_torch.pth \
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
  --ref_model_l2_weight 0.0 \
  --online_feature_top_fraction 1.0 \
  --no_use_lora
