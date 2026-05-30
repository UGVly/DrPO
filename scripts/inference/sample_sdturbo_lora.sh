#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Usage: scripts/inference/sample_sdturbo_lora.sh /path/to/checkpoint" >&2
  exit 2
fi

checkpoint=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src

python -m inference.sd_turbo_lora \
  --checkpoint "$checkpoint" \
  --outputs-dir "$PROJECT_ROOT"/outputs \
  --pretrained-model-path "$PROJECT_ROOT"/models/sd-turbo \
  --prompt-file "$PROJECT_ROOT"/data/pickscore/test.txt \
  --samples-dir "$PROJECT_ROOT"/samples \
  --seeds 42,43,44,45,46 \
  --batch-size 4 \
  --resolution 512 \
  --num-inference-steps 1 \
  --guidance-scale 0.0 \
  --device cuda \
  --dtype fp16 \
  --overwrite
