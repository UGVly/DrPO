# Source and Script Layout

This repository now separates reusable DrPO library code from comparison baselines.

```text
src/drpo/
  training/  # DrPO LoRA/full training entrypoints and shared trainer
  methods/   # main-method shims and SDXL trainers

baselines/
  draft/
  dpo/
  grpo/
  neighbor_grpo/
  spo/
  vggflow/
```

The matching script and output layout is:

```text
scripts/train/<method>.sh
scripts/experiments/<method>/...
outputs/<method>/...
```

## Canonical Entrypoints

Use these for new runs:

```bash
bash scripts/train/drpo.sh online
bash scripts/train/drpo.sh offline
bash scripts/train/drpo.sh offline_distance
bash scripts/train/drpo_full.sh online
bash scripts/train/draft.sh
bash scripts/train/dpo.sh online
bash scripts/train/grpo.sh
bash scripts/train/neighbor_grpo.sh
bash scripts/train/spo.sh
```

The `scripts/experiments/<method>/...` wrappers set common variants and then call the matching `scripts/train/<method>.sh` entrypoint.

## Baseline Implementations

Comparison algorithms live outside `src` under `baselines/<method>/`. Each baseline is LoRA-only and keeps its implementation to one or two Python files:

```text
baselines/draft/train_lora.py
baselines/dpo/train_lora.py
baselines/grpo/train_lora.py
baselines/neighbor_grpo/train_lora.py
baselines/neighbor_grpo/losses.py
baselines/spo/train_lora.py
baselines/spo/losses.py
baselines/vggflow/train_lora.py
baselines/vggflow/reward_gradient.py
```

The `src/drpo/methods/<baseline>/...` modules are compatibility shims only. New code should import reusable utilities from `src/drpo` and run baseline training through `scripts/train/*.sh` or the `baselines/<method>/train_lora.py` scripts directly.

## Legacy Compatibility

Main DrPO implementations live under `src/drpo/training`. Comparison baseline implementations live under `baselines`.

```text
Canonical                           Legacy
scripts/train/drpo.sh               scripts/drpo/train_drpo_sdturbo_lora.sh
scripts/train/draft.sh              scripts/draft/train_draft_sdturbo_lora.sh
scripts/train/dpo.sh                baselines/dpo/train_lora.py
scripts/train/grpo.sh               baselines/grpo/train_lora.py
scripts/train/neighbor_grpo.sh      no legacy equivalent
scripts/train/spo.sh                no legacy equivalent
```

The canonical DPO and GRPO wrappers write to `outputs/dpo/...` and `outputs/grpo/...`.

## LoRA and Full DrPO Entrypoints

SD-Turbo DrPO has two explicit Python entrypoints:

```text
src/drpo/training/sdturbo_lora.py
src/drpo/training/sdturbo_full.py
```

The LoRA entrypoint always trains LoRA adapters. The full entrypoint always trains the full UNet. Both reuse the same compact trainer implementation in `src/drpo/training/trainer.py` so the training logic is not over-split.

## DrPO Ablations

The detailed online DrPO ablation matrix remains under `scripts/drpo/online/` because those wrappers are already specific and stable:

```text
batch_gen_ablation/
different_reward/
feature_ablation/
kernel_ablation/
prompt_dataset/
reference_ablation/
selection_ablation/
weight_ablation/
```

These wrappers eventually call `scripts/drpo/train_drpo_sdturbo_lora.sh online`. They remain valid, and the canonical train wrapper also delegates to the same DrPO training script.

## SPO

SPO has a one-step SD-Turbo trainer adapted from Pairwise Sample Optimization. Adjacent generated samples are compared by reward and optimized with a pairwise log-ratio objective against the frozen reference model.

The default SPO run is:

```bash
bash scripts/experiments/spo/default.sh
```

It writes to:

```text
outputs/spo/<choice_model>/<regularization>/<timestamp>/
```

## Inference Layout

Inference utilities live under `src/inference/`, with shell wrappers in `scripts/inference/`:

```text
scripts/inference/sample_sdturbo_baseline.sh
scripts/inference/sample_sdturbo_lora.sh
scripts/inference/sample_all_checkpoints_5880.sh
scripts/inference/evaluate_samples.sh
```

Generated samples are stored under `samples/`, and metrics are stored under `samples/metrics/`. See `docs/inference_and_evaluation.md` for the sampling and metric workflow.
