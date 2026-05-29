#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"/src

ARGS=(
  --metric-set core \
  --samples-dir "$PROJECT_ROOT"/samples \
  --metrics-dir "$PROJECT_ROOT"/samples/metrics \
  --device cuda \
  --reward-batch-size 8 \
  --components pickscore,clip_aes,hpsv2,summarize
)

if [[ -n "${MANIFEST:-}" ]]; then
  ARGS+=(--manifest "$MANIFEST")
fi
if [[ -n "${MANIFEST_LIST:-}" ]]; then
  ARGS+=(--manifest-list "$MANIFEST_LIST")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

python -m inference.metrics "${ARGS[@]}"

if [[ "${RUN_IMAGEREWARD:-0}" == "1" ]]; then
  python -m inference.metrics "${ARGS[@]/core/imagereward}"
fi
