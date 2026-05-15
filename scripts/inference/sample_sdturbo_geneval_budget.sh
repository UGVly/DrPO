#!/usr/bin/env bash
set -euo pipefail

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO
export PYTHONPATH=/datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src

python -m inference.sd_turbo_geneval_budget \
  --pretrained-model-path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/sd-turbo \
  --prompt-file /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/third_party/geneval/prompts/evaluation_metadata.jsonl \
  --samples-dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/samples \
  --batch-size 8 \
  --resolution 512 \
  --num-inference-steps 1 \
  --guidance-scale 0.0 \
  --device cuda \
  --dtype fp16 \
  --run-name geneval_budget_base \
  --max-attempts 30 \
  --seed-base 42 \
  --geneval-repo /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/third_party/geneval \
  --geneval-detector-path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/geneval_detector
