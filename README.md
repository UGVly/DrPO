# Drifting Preference Optimization

Drifting Preference Optimization (DrPO) provides compact training, inference,
and evaluation code for preference optimization on one-step text-to-image
diffusion models.

The repository is intentionally small:

```text
src/drpo/          Core DrPO library, rewards, feature extraction, trainers
src/inference/     Sampling and metric utilities
baselines/         Compact comparison methods used in the paper
scripts/train/     Reproducible launch recipes
scripts/inference/ Sampling and evaluation wrappers
tests/             Layout and numerical unit tests
```

Local datasets, checkpoints, logs, generated figures, paper builds, and one-off
cluster launch scripts are excluded from the open-source tree.

## Environment

This repo is maintained with conda.

```bash
conda env create -f environment.yml
conda activate drpo
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
models/sd-turbo/
models/sdxl-turbo/
models/PickScore_v1/
models/CLIP-ViT-L-14/
models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
models/HPSv2/HPS_v2_compressed.pt
models/Aesthetic/sac+logos+ava1-l14-linearMSE.pth
models/mae_latent_256_torch.pth
data/prompts/pickapicv2_test_unique.txt
data/pairs.jsonl
```

Paths can be overridden with environment variables such as
`PRETRAINED_MODEL_PATH`, `PICKSCORE_MODEL_PATH`, `HPS_CKPT_PATH`,
`AESTHETIC_CKPT_PATH`, `HPS_OPEN_CLIP_PRETRAINED_PATH`, and `PROMPT_FILE`.

Download the latent MAE checkpoint from ModelScope:

```bash
python -m pip install modelscope
mkdir -p models
modelscope download \
  --model jiangzhou130v1/drpo-mae-latent-256 \
  mae_latent_256_torch.pth \
  --local_dir models
```

The file is written to `models/mae_latent_256_torch.pth`. Its SHA256 checksum
is `4810249905d2882a41d5a0fe97ebac995af8918dbe121daa9871e5bc605445b1`.

## SDXL-Turbo Runs

SDXL-Turbo MAE-DrPO, teacher U-Net DrPO, and Draft wrappers default to
`LEARNING_RATE=1e-5`.
The DrPO path has two feature backends: pixel MAE and frozen teacher U-Net hidden states.

```bash
conda activate drpo
bash scripts/train/sdxl_turbo_drpo_mae.sh
bash scripts/train/sdxl_turbo_drpo_teacher.sh
bash scripts/train/sdxl_turbo_draft.sh
```

## Repository Hygiene

For release preparation, keep only reusable code under `src/`, maintained launch
recipes under `scripts/`, and tests/docs that describe those paths. Put local
sweeps, paper-rendering artifacts, copied PDFs, generated figures, and machine
specific launch helpers outside the tracked tree.
