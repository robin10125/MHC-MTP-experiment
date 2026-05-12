#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-runs_wikitext103_fullpass_1024x14_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-14339}"  # One WikiText-103 tokenized pass at 8192 tokens/update.

COMMON=(
  --profile rtx2070
  --dataset wikitext103
  --d_model 1024
  --n_layers 14
  --n_heads 16
  --batch_size 2
  --grad_accum 16
  --seq_len 256
  --n_steps "$STEPS"
  --val_every 500
  --save_every 5000
  --log_every 100
  --val_batches 20
  --run_dir "$RUN_DIR"
)

run_one() {
  local label="$1"
  shift
  echo "=== ${label} WikiText-103 full-pass run ==="
  .venv/bin/python train.py "${COMMON[@]}" "$@"
}

echo "Run dir: ${RUN_DIR}"
echo "Steps per baseline: ${STEPS}"
echo "Effective tokens per step: 8192"
echo "Approx tokens per baseline: $((STEPS * 8192))"

run_one residual_only --model residual_only
run_one mhc_only --model mhc_only
run_one residual_mtp --model residual_mtp
run_one mhc_mtp_sum --model mhc_mtp --reduction sum
run_one mhc_mtp_mix --model mhc_mtp --reduction mix
