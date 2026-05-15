#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Usage: scripts/inference/sample_sdturbo_lora.sh /path/to/checkpoint" >&2
  exit 2
fi

checkpoint=$1

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO
export PYTHONPATH=/datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src

python -m inference.sd_turbo_lora \
  --checkpoint "$checkpoint" \
  --outputs-dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/outputs \
  --pretrained-model-path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/sd-turbo \
  --prompt-file /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/data/prompts/pickapicv2_test_unique.txt \
  --samples-dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/samples \
  --seeds 42,43,44,45,46 \
  --batch-size 4 \
  --resolution 512 \
  --num-inference-steps 1 \
  --guidance-scale 0.0 \
  --device cuda \
  --dtype fp16 \
  --overwrite
