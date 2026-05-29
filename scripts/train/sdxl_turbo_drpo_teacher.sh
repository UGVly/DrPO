#!/usr/bin/env bash
set -euo pipefail

# TUNE_TRACE 2026-05-17:
#   Default SDXL-Turbo + Teacher-UNet DrPO candidate for score/diversity tuning.
#   Keep optimization settings fixed at lr=1e-5, max_train_steps=5000,
#   batchsize_gen=24, bf16, LoRA rank/alpha=16. Change only launch width
#   (num_processes=4, gradient_accumulation_steps=8) and loss-side knobs.
#   This "balanced" setting mildly lowers preference-negative pressure and
#   strengthens reference/diversity preservation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENV_BIN="${ENV_BIN:-}"
if [ -n "$ENV_BIN" ]; then
  export PATH="$ENV_BIN:$PATH"
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export TOKENIZERS_PARALLELISM=false

RUN_NAME="${RUN_NAME:-balanced_refdiv_lr1e-5_bs24_pos12_neg12_ga8_steps5000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/sdxl-turbo-lora/drpo/teacher_unet/tuning/$RUN_NAME}"
mkdir -p "$OUTPUT_DIR"
cat > "$OUTPUT_DIR/tune_trace.txt" <<TRACE
variant=balanced_refdiv
date=2026-05-17
intent=balance PickScore improvement with diversity/reference preservation
fixed_optimization=learning_rate=1e-5,max_train_steps=5000,batchsize_gen=24,mixed_precision=bf16,lora_r=16,lora_alpha=16
launch=num_processes=${NUM_PROCESSES:-4},gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-8}
loss=pos=3000,neg=2000,ref=3500,ref_neg=5000,ref_loss_weight=0.25,ref_model_l2=0.03,feature_diversity_weight=1.0,feature_diversity_margin_scale=0.80
TRACE

accelerate launch \
  --num_processes "${NUM_PROCESSES:-4}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29642}" \
  "$PROJECT_ROOT/src/drpo/methods/sdxl_drpo/trainer.py" \
  --pretrained_model_name_or_path "$PROJECT_ROOT/models/stable-diffusion-xl-turbo" \
  --prompt_file "$PROJECT_ROOT/data/prompts/pickapicv2_test_unique.txt" \
  --output_dir "$OUTPUT_DIR" \
  --feature_extractor teacher_unet \
  --teacher_feature_layers down_blocks.2,mid_block,up_blocks.0 \
  --teacher_feature_noise 0.1 \
  --teacher_feature_timestep 100 \
  --teacher_feature_pool_size 4 \
  --choice_model pickscore \
  --pickscore_model_name_or_path "$PROJECT_ROOT/models/PickScore_v1" \
  --pickscore_processor_name_or_path "$PROJECT_ROOT/models/PickScore_v1" \
  --resolution 512 \
  --batchsize_gen 24 \
  --num_pos_images 12 \
  --num_neg_images 12 \
  --max_train_steps 5000 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --checkpointing_steps 100 \
  --drifting_pos_weight 3000.0 \
  --drifting_neg_weight 2000.0 \
  --drifting_ref_weight 3500.0 \
  --drifting_ref_neg_weight 5000.0 \
  --drifting_ref_loss_weight 0.25 \
  --ref_model_l2_weight 0.03 \
  --feature_diversity_weight 1.0 \
  --feature_diversity_margin_scale 0.80 \
  --use_lora
