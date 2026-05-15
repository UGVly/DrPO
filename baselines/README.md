# Comparison Baselines

This directory contains LoRA-only comparison algorithms outside the reusable `src/drpo` package.

Each baseline keeps its training implementation self-contained. The GRPO baseline is the fused neighbor-based variant; the older plain GRPO and Neighbor-GRPO paths are no longer separate algorithms.

```text
draft/train_lora.py
dpo/train_lora.py
grpo/train_lora.py
grpo/trainer.py
grpo/losses.py
spo/train_lora.py
spo/losses.py
vggflow/train_lora.py
vggflow/reward_gradient.py
```

Use the matching `scripts/train/<method>.sh` wrapper for normal launches.
