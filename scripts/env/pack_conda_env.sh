#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ $# -gt 0 ]; then
  env_name=$1
else
  env_name=strong-drpo
fi

out_dir="$PROJECT_ROOT"/conda_venvs
out_path=$out_dir/$env_name.tar.gz
mkdir -p "$out_dir"

if ! command -v conda-pack >/dev/null 2>&1; then
  echo "conda-pack is required in the active environment." >&2
  exit 1
fi

conda-pack -n "$env_name" -o "$out_path" --force
echo "$out_path"
