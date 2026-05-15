#!/usr/bin/env bash
set -euo pipefail

cd /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO

python scripts/qwen_vl_eval/pairwise_winrate.py \
  --sample-root /datapool \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --torch-dtype bfloat16 \
  --device-map auto \
  --output-dir /datapool/jiangzhou/CODE/Text2ImageProject/StrongDrPO/analysis/qwen_vl_winrate \
  --run-name draft_drpo_qwen3vl
