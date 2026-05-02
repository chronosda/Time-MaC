#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/chronos/Time-me"
DATA_ROOT="${ROOT}/dataset/electricity"
LOG_DIR="${ROOT}/logs"
CKPT_DIR="${ROOT}/checkpoints"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

COMMON_ARGS=(
  --root_path "${DATA_ROOT}"
  --data electricity
  --seq_len 512
  --pred_len 96
  --d_model 256
  --batch_size 6
  --epochs 20
  --learning_rate 1e-4
  --vlm_type mae
  --offline
  --mae_load_ckpt
  --use_gpu
  --gpu 0
  --val_ratio 0.1
  --calib_ratio 0.1
  --test_ratio 0.1
)

run_one() {
  local tag="$1"
  local subset="$2"
  shift 2

  echo "============================================================"
  echo "Running ${tag} with train_subset_ratio=${subset}"
  echo "============================================================"

  python "${ROOT}/scripts/train_time_me.py" \
    "${COMMON_ARGS[@]}" \
    --save_path "${CKPT_DIR}/${tag}" \
    --train_ratio 0.7 \
    --train_subset_ratio "${subset}" \
    "$@" \
    > "${LOG_DIR}/${tag}.log" 2>&1
}

for subset in 0.2 0.4; do
  run_one "electricity_p96_mae_std_no_cm_lowres_${subset}" "${subset}"
  run_one "electricity_p96_mae_std_cm_lowres_${subset}" "${subset}" --use_enhanced_fusion
done

echo "Low-resource ablation runs finished."
