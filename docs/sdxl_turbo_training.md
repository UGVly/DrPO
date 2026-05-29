# SDXL-Turbo Training

This repo keeps the maintained SDXL-Turbo entrypoints small and reproducible.
All paths are relative to the repository root.

## Entrypoints

```bash
bash scripts/train/sdxl_turbo_drpo_mae.sh
bash scripts/train/sdxl_turbo_drpo_teacher.sh
bash scripts/train/sdxl_turbo_draft.sh
bash scripts/train/sdxl_turbo_grpo.sh
```

The matching trainers live in:

```text
src/drpo/methods/sdxl_drpo/trainer.py
src/drpo/methods/sdxl_draft/trainer.py
src/drpo/methods/sdxl_grpo/trainer.py
```

## Default Recipe

The maintained SDXL recipes use:

```text
pretrained_model_name_or_path = models/stable-diffusion-xl-turbo
prompt_file = data/prompts/pickapicv2_test_unique.txt
choice_model = pickscore
mixed_precision = bf16
batchsize_gen = 16
num_pos_images = 8
num_neg_images = 8
max_train_steps = 5000
use_lora = true
lora_r = 16
lora_alpha = 16
lora_target_modules = to_q,to_k,to_v,to_out.0
```

`sdxl_turbo_drpo_mae.sh` uses MAE features. `sdxl_turbo_drpo_teacher.sh` uses
frozen teacher U-Net hidden states from `down_blocks.2`, `mid_block`, and
`up_blocks.0`.

Machine-specific sweep launchers and paper-table scripts are intentionally not
part of the tracked open-source tree.
