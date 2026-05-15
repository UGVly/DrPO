#!/usr/bin/env bash
set -euo pipefail

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO
export PYTHONPATH=/datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/src

python -m inference.metrics \
  --metric-set core \
  --samples-dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/samples \
  --metrics-dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/samples/metrics \
  --device cuda \
  --reward-batch-size 8 \
  --feature-batch-size 16 \
  --fid-batch-size 32 \
  --dino-model-path /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/models/dinov2-base
