#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src

python -m inference.sd_turbo_geneval_budget \
  --pretrained-model-path "$PROJECT_ROOT"/models/sd-turbo \
  --prompt-file "$PROJECT_ROOT"/third_party/geneval/prompts/evaluation_metadata.jsonl \
  --samples-dir "$PROJECT_ROOT"/samples \
  --batch-size 8 \
  --resolution 512 \
  --num-inference-steps 1 \
  --guidance-scale 0.0 \
  --device cuda \
  --dtype fp16 \
  --run-name geneval_budget_base \
  --max-attempts 30 \
  --seed-base 42 \
  --geneval-repo "$PROJECT_ROOT"/third_party/geneval \
  --geneval-detector-path "$PROJECT_ROOT"/models/geneval_detector
