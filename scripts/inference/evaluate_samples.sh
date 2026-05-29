#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src

python -m inference.metrics \
  --metric-set core \
  --samples-dir "$PROJECT_ROOT"/samples \
  --metrics-dir "$PROJECT_ROOT"/samples/metrics \
  --device cuda \
  --reward-batch-size 8 \
  --feature-batch-size 16 \
  --fid-batch-size 32 \
  --dino-model-path "$PROJECT_ROOT"/models/dinov2-base
