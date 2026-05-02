#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT}/dataset"
LOG_DIR="${ROOT}/logs"
CKPT_DIR="${ROOT}/checkpoints"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

run_one() {
  local pred_len="$1"
  local tag="$2"

  echo "[ETTm2] starting ${tag} (pred_len=${pred_len})"
  python "${ROOT}/scripts/train_time_me.py" \
    --data ETTm2 \
    --root_path "${DATA_ROOT}" \
    --save_path "${CKPT_DIR}/${tag}" \
    --pred_len "${pred_len}" \
    --seq_len 512 \
    --d_model 512 \
    --batch_size 16 \
    --epochs 15 \
    --learning_rate 0.001 \
    --num_workers 4 \
    --use_gpu \
    --gpu 0 \
    --vlm_type clip \
    --conformal_enable \
    > "${LOG_DIR}/${tag}.log" 2>&1
}

run_one 96  "ettm2-time-me-p96-rerun"
run_one 192 "ettm2-time-me-p192-rerun"
run_one 336 "ettm2-time-me-p336-rerun"
run_one 720 "ettm2-time-me-p720-rerun"
