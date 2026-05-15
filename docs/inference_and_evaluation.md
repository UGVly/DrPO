# Inference and Evaluation

This document covers the post-training workflow for SD-Turbo baseline sampling, LoRA checkpoint sampling, and offline metrics.

## Inputs

Sampling reads prompts from:

```text
data/prompts/pickapicv2_test_unique.txt
```

Override with `PROMPT_FILE=/path/to/prompts.txt`. Each non-empty line is one prompt. Sampling uses seeds `42,43,44,45,46` by default; override with `SEEDS=42,43`.

The default SD-Turbo base model path is:

```text
models/sd-turbo/
```

Override with `PRETRAINED_MODEL_PATH=/path/to/sd-turbo`.

## Baseline Sampling

Generate the base SD-Turbo one-step baseline:

```bash
bash scripts/inference/sample_sdturbo_baseline.sh
```

Useful overrides:

```bash
MAX_PROMPTS=100 \
SEEDS=42,43,44 \
BATCH_SIZE=8 \
RUN_NAME=pickapicv2_100 \
bash scripts/inference/sample_sdturbo_baseline.sh
```

Outputs:

```text
samples/sd-turbo-baseline/<run_name>/
  manifest.jsonl
  seed_<seed>/images/<prompt_id>.png
```

## LoRA Checkpoint Sampling

Sample a single checkpoint:

```bash
CHECKPOINT_PATH=outputs/drpo/online/default/<timestamp>/checkpoint-100 \
bash scripts/inference/sample_sdturbo_lora.sh
```

The checkpoint can be one of:

```text
checkpoint-100/
checkpoint-100/unet_lora/
checkpoint-100/unet_lora/adapter_model.safetensors
```

By default, the output path mirrors the checkpoint path relative to `outputs/`:

```text
samples/sd-turbo-lora/drpo/online/default/<timestamp>/checkpoint-100/
  manifest.jsonl
  seed_<seed>/images/<prompt_id>.png
```

Useful overrides:

```bash
CHECKPOINT_PATH=/path/to/checkpoint-100 \
OUTPUT_DIR=samples/manual/drpo-checkpoint-100 \
MAX_PROMPTS=100 \
FORCE=1 \
bash scripts/inference/sample_sdturbo_lora.sh
```

`FORCE=1` resamples existing image files.

## Batch Checkpoint Sampling

Discover all LoRA checkpoints under `outputs/`:

```bash
uv run python -m inference.discover --outputs-dir outputs --print-paths
```

Run baseline plus every discovered checkpoint, one job per idle GPU:

```bash
GPUS=0,1,2,3 bash scripts/inference/sample_all_checkpoints_5880.sh
```

Useful overrides:

```bash
RUN_BASELINE=0 \
MAX_CHECKPOINTS=8 \
GPU_MAX_MEMORY_MB=1024 \
GPU_MAX_UTIL=10 \
SCHEDULER_POLL_SECONDS=10 \
bash scripts/inference/sample_all_checkpoints_5880.sh
```

To use an explicit checkpoint list:

```bash
CHECKPOINT_LIST=/path/to/checkpoints.txt bash scripts/inference/sample_all_checkpoints_5880.sh
```

Each non-empty line in `CHECKPOINT_LIST` should be a checkpoint directory or adapter path.

## Core Metrics

Run the default metric workflow:

```bash
bash scripts/inference/evaluate_samples.sh
```

Core metrics compute:

```text
pickscore
clip
aes
hpsv2
clip_diversity
dino_diversity
fid_vs_baseline
```

Metrics are written to:

```text
samples/metrics/
  summary.csv
  <sample-relative-path>/scores.jsonl
  <sample-relative-path>/summary.json
```

`fid_vs_baseline` uses `samples/sd-turbo-baseline/default/manifest.jsonl` unless `--baseline-manifest` is passed directly to `python -m inference.metrics`.

Evaluate one manifest:

```bash
MANIFEST=samples/sd-turbo-lora/drpo/online/default/<timestamp>/checkpoint-100/manifest.jsonl \
bash scripts/inference/evaluate_samples.sh
```

Recompute existing metric files:

```bash
FORCE=1 bash scripts/inference/evaluate_samples.sh
```

Skip ImageReward:

```bash
RUN_IMAGEREWARD=0 bash scripts/inference/evaluate_samples.sh
```

## ImageReward Metrics

ImageReward is run in an isolated uv project because its dependency stack is different from the training environment:

```bash
uv sync --project eval_envs/imagereward
RUN_IMAGEREWARD=1 bash scripts/inference/evaluate_samples.sh
```

The ImageReward pass reuses existing `scores.jsonl` when present, appends `imagereward`, updates `summary.json`, and refreshes `summary.csv`.

## Direct Python Entrypoints

The shell wrappers set `PYTHONPATH` and choose a Python binary automatically. The underlying modules are:

```bash
python -m inference.sd_turbo_baseline
python -m inference.sd_turbo_lora --checkpoint /path/to/checkpoint
python -m inference.discover --outputs-dir outputs --print-paths
python -m inference.metrics --metric-set core
python -m inference.metrics --metric-set imagereward
```

For quick smoke runs, combine `MAX_PROMPTS` and a short seed list:

```bash
MAX_PROMPTS=4 SEEDS=42 BATCH_SIZE=2 bash scripts/inference/sample_sdturbo_baseline.sh
RUN_IMAGEREWARD=0 MANIFEST=samples/sd-turbo-baseline/default/manifest.jsonl bash scripts/inference/evaluate_samples.sh
```
