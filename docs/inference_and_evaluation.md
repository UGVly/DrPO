# Inference and Evaluation

This document covers the post-training workflow for SD-Turbo baseline sampling, LoRA checkpoint sampling, and offline metrics.

## Inputs

The shell wrappers sample the default prompt file:

```text
data/prompts/pickapicv2_test_unique.txt
```

Each non-empty line is one prompt. The wrappers use seeds `42,43,44,45,46`.
Use the Python entrypoints directly when you need custom prompt files, seeds,
batch sizes, or output directories.

The default SD-Turbo base model path is:

```text
models/sd-turbo/
```
Use `--pretrained-model-path /path/to/sd-turbo` with the Python entrypoints to
override it.

## Baseline Sampling

Generate the base SD-Turbo one-step baseline:

```bash
bash scripts/inference/sample_sdturbo_baseline.sh
```

Equivalent direct command with common overrides:

```bash
python -m inference.sd_turbo_baseline \
  --prompt-file data/prompts/pickapicv2_test_unique.txt \
  --seeds 42,43,44 \
  --max-prompts 100 \
  --batch-size 8 \
  --run-name pickapicv2_100
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
bash scripts/inference/sample_sdturbo_lora.sh outputs/draft/sdturbo_lora/pickscore_default/checkpoint-100
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

Equivalent direct command with common overrides:

```bash
python -m inference.sd_turbo_lora \
  --checkpoint /path/to/checkpoint-100 \
  --output-dir samples/manual/drpo-checkpoint-100 \
  --max-prompts 100 \
  --overwrite
```

`--overwrite` resamples existing image files.

Discover all LoRA checkpoints under `outputs/`:

```bash
python -m inference.discover --outputs-dir outputs --print-paths
```

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
```

The compact evaluator intentionally keeps only reward scores. Keep broader
analysis in separate notebooks or downstream evaluation repos if needed.

Metrics are written to:

```text
samples/metrics/
  summary.csv
  <sample-relative-path>/scores.jsonl
  <sample-relative-path>/summary.json
```

Evaluate one manifest:

```bash
MANIFEST=samples/sd-turbo-lora/drpo/online/default/<timestamp>/checkpoint-100/manifest.jsonl \
bash scripts/inference/evaluate_samples.sh
```

Recompute existing metric files:

```bash
FORCE=1 bash scripts/inference/evaluate_samples.sh
```

## Direct Python Entrypoints

The shell wrappers set `PYTHONPATH` and choose a Python binary automatically. The underlying modules are:

```bash
python -m inference.sd_turbo_baseline
python -m inference.sd_turbo_lora --checkpoint /path/to/checkpoint
python -m inference.discover --outputs-dir outputs --print-paths
python -m inference.metrics --metric-set core
```

For quick smoke runs, combine `MAX_PROMPTS` and a short seed list:

```bash
python -m inference.sd_turbo_baseline --max-prompts 4 --seeds 42 --batch-size 2
MANIFEST=samples/sd-turbo-baseline/default/manifest.jsonl bash scripts/inference/evaluate_samples.sh
```
