# StrongDrPO

StrongDrPO is a cleaned-up DrPO/Draft training repo derived from `/datapool/jiangzhou/CODE/Text2ImageProject/DrPO`.
It keeps the research code in `src/`, lightweight launch wrappers in `scripts/`, and excludes local datasets, checkpoints, logs, and packed environments from git.

## Environment

This repo is maintained with conda.

```bash
conda env create -f environment.yml
conda activate strong-drpo
pip install -e ".[dev]"
```

To pack an existing environment for another machine:

```bash
bash scripts/env/pack_conda_env.sh mydiffusers
```

The archive is written to `conda_venvs/<env>.tar.gz`.

## Local Assets

Training expects local assets under the project root:

```text
models/stable-diffusion-xl-turbo/
models/facebook-vit-mae-base/
models/PickScore_v1/
data/prompts/pickapicv2_test_unique.txt
data/pairs.jsonl
```

Paths can be overridden with environment variables such as `PRETRAINED_MODEL_PATH`, `MAE_MODEL_PATH`, `PICKSCORE_MODEL_PATH`, and `PROMPT_FILE`.

## SDXL-Turbo Runs

SDXL-Turbo MAE-DrPO and Draft wrappers default to `LEARNING_RATE=1e-5`.
The teacher U-Net feature DrPO wrapper defaults to `LEARNING_RATE=1e-6`.
The DrPO path has two feature backends: pixel MAE and frozen teacher U-Net hidden states.

```bash
conda activate strong-drpo
bash scripts/train/sdxl_turbo_drpo_mae.sh
bash scripts/train/sdxl_turbo_drpo_teacher.sh
bash scripts/train/sdxl_turbo_draft.sh
```

For `wyd-h100`, use:

```bash
bash scripts/manual_launches/run_sdxl_turbo_drpo_draft_wyd_h100_lr1e-5.sh
```

The launcher uses the active conda installation, activates `CONDA_ENV_NAME` (default `mydiffusers`), and writes logs under `logs/launcher_logs/`.

## SD-Turbo DrPO Entrypoints

LoRA and full-UNet DrPO training use explicit entrypoints:

```bash
bash scripts/train/drpo.sh online
bash scripts/train/drpo_full.sh online
```

Both entrypoints share `src/drpo/training/trainer.py`; only the adapter/full training mode is fixed by the entrypoint.
