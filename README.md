# Drifting Preference Optimization

Official implementation of **Drifting Preference Optimization (DrPO)** for
preference optimization of one-step text-to-image diffusion models.

This release contains compact training code for SDXL-Turbo and SD-Turbo,
PickScore-based prompt splits, baseline comparisons, and sampling/evaluation
utilities.

## News

- The default training split is `data/pickscore/train.txt`.
- The default test/evaluation split is `data/pickscore/test.txt`.
- SDXL-Turbo DrPO supports MAE features and teacher U-Net features.

## Project Layout

```text
src/drpo/          Core DrPO package and SDXL-Turbo trainers
src/inference/     Sampling and metric utilities
baselines/         SD-Turbo comparison methods
scripts/train/     Reproducible training launch scripts
scripts/inference/ Sampling and evaluation wrappers
data/pickscore/    Prompt splits used by the default scripts
tests/             Lightweight layout and configuration tests
```

## Installation

```bash
git clone https://github.com/UGVly/DrPO.git
cd DrPO

conda env create -f environment.yml
conda activate drpo
pip install -e .
```

The code expects model weights to be available locally. It does not download
large checkpoints during training.

## Model Weights

Create the expected local model directories:

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

modelscope download \
  --model jiangzhou130v1/drpo-mae-latent-256 \
  mae_latent_256_torch.pth \
  --local_dir models
```

If a Hugging Face model requires authentication, run `huggingface-cli login`
and repeat the command.

Check the default assets:

```bash
python scripts/check_local_assets.py
```

Expected default paths:

```text
models/sd-turbo/
models/stable-diffusion-xl-turbo/
models/facebook-vit-mae-base/
models/PickScore_v1/
models/mae_latent_256_torch.pth
data/pickscore/train.txt
data/pickscore/test.txt
```

## Training

SDXL-Turbo DrPO with teacher U-Net features:

```bash
bash scripts/train/sdxl_turbo_drpo_teacher.sh
```

SDXL-Turbo DrPO with MAE features:

```bash
bash scripts/train/sdxl_turbo_drpo_mae.sh
```

SDXL-Turbo Draft baseline:

```bash
bash scripts/train/sdxl_turbo_draft.sh
```

SD-Turbo comparison baselines:

```bash
bash scripts/train/draft.sh
bash scripts/train/dpo.sh
bash scripts/train/grpo.sh
bash scripts/train/spo.sh
bash scripts/train/vggflow.sh
```

The shell wrappers are normal `accelerate launch` commands. Edit
`--num_processes`, batch sizes, or output directories in the wrapper scripts to
match your hardware.

## Sampling

Sample the SD-Turbo baseline:

```bash
bash scripts/inference/sample_sdturbo_baseline.sh
```

Sample an SD-Turbo LoRA checkpoint:

```bash
bash scripts/inference/sample_sdturbo_lora.sh outputs/draft/sdturbo_lora/pickscore_default/checkpoint-100
```

Sample an SDXL-Turbo LoRA checkpoint:

```bash
python -m inference.sdxl_turbo_lora \
  --checkpoint outputs/sdxl-turbo-lora/drpo/teacher_unet/default/checkpoint-100 \
  --prompt-file data/pickscore/test.txt \
  --pretrained-model-path models/stable-diffusion-xl-turbo \
  --device cuda
```

Generated images and manifests are written under `samples/` by default.

## Evaluation

Evaluate generated sample manifests with the core metric pipeline:

```bash
MANIFEST=samples/path/to/manifest.jsonl bash scripts/inference/evaluate_samples.sh
```

Score a custom JSONL file or a single image with a reward model:

```bash
python scripts/evaluate_rewards.py \
  --selector pickscore \
  --prompt "a cinematic photo of a red sports car" \
  --image path/to/image.png
```

## Optional Assets

The default training scripts only require SD-Turbo, SDXL-Turbo, MAE, and
PickScore. Additional reward selectors require extra local assets:

```text
models/CLIP-ViT-L-14/
models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
models/HPSv2/HPS_v2_compressed.pt
models/Aesthetic/sac+logos+ava1-l14-linearMSE.pth
```

Check optional assets with:

```bash
python scripts/check_local_assets.py --all
```

## Citation

If this code is useful for your research, please cite the DrPO paper.

```bibtex
@article{drpo2026,
  title   = {Drifting Preference Optimization},
  author  = {DrPO Authors},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

This repository is released for research use. Please check the licenses of the
underlying model checkpoints before using them.
