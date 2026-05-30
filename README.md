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

This repo is maintained with conda. The default launch scripts are intended for
CUDA machines and were tested on the lab H200 setup.

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

Training is local-only: the trainers do not download weights at runtime. Put the
default reproduction assets under the project root:

```text
models/sd-turbo/
models/stable-diffusion-xl-turbo/
models/facebook-vit-mae-base/
models/PickScore_v1/
data/pickscore/train.txt
data/pickscore/test.txt
```

Download them with:

```bash
python -m pip install -U "huggingface_hub[cli]" modelscope
mkdir -p models

huggingface-cli download stabilityai/sd-turbo \
  --local-dir models/sd-turbo

huggingface-cli download stabilityai/sdxl-turbo \
  --local-dir models/stable-diffusion-xl-turbo

huggingface-cli download facebook/vit-mae-base \
  --local-dir models/facebook-vit-mae-base

huggingface-cli download yuvalkirstain/PickScore_v1 \
  --local-dir models/PickScore_v1
```

If Hugging Face returns a gated-model or authentication error, run
`huggingface-cli login` and retry the same commands.

Check the default reproduction assets:

```bash
python scripts/check_local_assets.py
```

Optional reward and baseline assets are only needed for non-default reward
selectors such as `clip`, `aes`, `hps`, or the VGGFlow/aesthetic paths:

```text
models/CLIP-ViT-L-14/
models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
models/HPSv2/HPS_v2_compressed.pt
models/Aesthetic/sac+logos+ava1-l14-linearMSE.pth
models/mae_latent_256_torch.pth
```

Download the optional assets with:

```bash
huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir models/CLIP-ViT-L-14

huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  open_clip_pytorch_model.bin \
  --local-dir models/CLIP-ViT-H-14-laion2B-s32B-b79K

huggingface-cli download xswu/HPSv2 \
  HPS_v2_compressed.pt \
  --repo-type space \
  --local-dir models/HPSv2

huggingface-cli download camenduru/improved-aesthetic-predictor \
  "sac+logos+ava1-l14-linearMSE.pth" \
  --local-dir models/Aesthetic
```

Download the latent MAE checkpoint from ModelScope:

```bash
mkdir -p models
modelscope download \
  --model jiangzhou130v1/drpo-mae-latent-256 \
  mae_latent_256_torch.pth \
  --local_dir models
```

The file is written to `models/mae_latent_256_torch.pth`. Its SHA256 checksum
is `4810249905d2882a41d5a0fe97ebac995af8918dbe121daa9871e5bc605445b1`.

Check every optional asset too:

```bash
python scripts/check_local_assets.py --all
```

Paths can be overridden with environment variables such as
`PRETRAINED_MODEL_PATH`, `PICKSCORE_MODEL_PATH`, `HPS_CKPT_PATH`,
`AESTHETIC_CKPT_PATH`, `HPS_OPEN_CLIP_PRETRAINED_PATH`, and `PROMPT_FILE`.
HPSv3 and Qwen2-VL evaluation weights are not part of this release.

## SDXL-Turbo Runs

SDXL-Turbo MAE-DrPO, teacher U-Net DrPO, and Draft wrappers default to
`LEARNING_RATE=1e-5`.
The DrPO path has two feature backends: pixel MAE and frozen teacher U-Net
hidden states. The scripts use the lab multi-GPU defaults in `scripts/train/`;
adjust `--num_processes` inside the wrapper if you run on fewer GPUs.

```bash
conda activate drpo
bash scripts/train/sdxl_turbo_drpo_mae.sh
bash scripts/train/sdxl_turbo_drpo_teacher.sh
bash scripts/train/sdxl_turbo_draft.sh
```

The SD-Turbo default baseline is:

```bash
conda activate drpo
bash scripts/train/draft.sh
```

## Repository Hygiene

For release preparation, keep only reusable code under `src/`, maintained launch
recipes under `scripts/`, and focused tests that cover those paths. Put local
sweeps, paper-rendering artifacts, copied PDFs, generated figures, and machine
specific launch helpers outside the tracked tree.
