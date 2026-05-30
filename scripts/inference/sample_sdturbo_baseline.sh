#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src

python -m inference.sd_turbo_baseline \
  --pretrained-model-path "$PROJECT_ROOT"/models/sd-turbo \
  --prompt-file "$PROJECT_ROOT"/data/pickscore/test.txt \
  --samples-dir "$PROJECT_ROOT"/samples \
  --run-name default \
  --seeds 42,43,44,45,46 \
  --batch-size 4 \
  --resolution 512 \
  --num-inference-steps 1 \
  --guidance-scale 0.0 \
  --device cuda \
  --dtype fp16 \
  --overwrite
