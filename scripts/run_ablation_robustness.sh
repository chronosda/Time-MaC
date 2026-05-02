#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/chronos/Time-me"
DATA_ROOT="${ROOT}/dataset/electricity"
LOG_DIR="${ROOT}/logs"

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --root_path "${DATA_ROOT}"
  --data electricity
  --seq_len 512
  --pred_len 96
  --d_model 256
  --batch_size 6
  --vlm_type mae
  --offline
  --mae_load_ckpt
  --use_gpu
  --gpu 0
  --train_ratio 0.7
  --val_ratio 0.1
  --calib_ratio 0.1
  --test_ratio 0.1
)

run_eval() {
  local tag="$1"
  local ckpt_dir="$2"
  shift 2

  python "${ROOT}/scripts/eval_time_me_robustness.py" \
    "${COMMON_ARGS[@]}" \
    --checkpoint_dir "${ckpt_dir}" \
    "$@" \
    > "${LOG_DIR}/${tag}.log" 2>&1
}

NO_CM="${ROOT}/checkpoints/electricity_p96_mae_std_no_cm_512"
CM="${ROOT}/checkpoints/electricity_p96_mae_std_cm_512"

run_eval "robustness_no_cm_clean" "${NO_CM}"
run_eval "robustness_cm_clean" "${CM}" --use_enhanced_fusion

run_eval "robustness_no_cm_noise005" "${NO_CM}" --eval_noise_std 0.05
run_eval "robustness_cm_noise005" "${CM}" --use_enhanced_fusion --eval_noise_std 0.05

run_eval "robustness_no_cm_noise01" "${NO_CM}" --eval_noise_std 0.1
run_eval "robustness_cm_noise01" "${CM}" --use_enhanced_fusion --eval_noise_std 0.1

run_eval "robustness_no_cm_mask01" "${NO_CM}" --eval_mask_ratio 0.1 --eval_mask_mode block --eval_mask_block_len 24
run_eval "robustness_cm_mask01" "${CM}" --use_enhanced_fusion --eval_mask_ratio 0.1 --eval_mask_mode block --eval_mask_block_len 24

run_eval "robustness_no_cm_mask02" "${NO_CM}" --eval_mask_ratio 0.2 --eval_mask_mode block --eval_mask_block_len 24
run_eval "robustness_cm_mask02" "${CM}" --use_enhanced_fusion --eval_mask_ratio 0.2 --eval_mask_mode block --eval_mask_block_len 24

echo "Robustness evaluations finished."
