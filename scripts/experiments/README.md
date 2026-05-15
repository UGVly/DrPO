# Experiment Scripts

Canonical experiment scripts are grouped by method so they mirror the maintained baselines and SDXL trainers.

- `drpo/`: main DrPO experiments and ablations.
- `draft/`: Draft baseline runs.
- `dpo/`: one-step DPO variants.
- `grpo/`: fused neighbor-based GRPO runs.
- `spo/`: SPO baseline slot.
- `vggflow/`: one-step VGG-Flow reward-gradient baseline.

Experiment wrappers should call the canonical entries in `scripts/train/`. Comparison method implementations live under `baselines/<method>/`; `src/drpo` is reserved for reusable DrPO library code and main trainers.
