#!/usr/bin/env bash
#
# Ablation script for Time-me on Electricity-p96:
# - Switch Coupled-Mamba on/off
# - Switch between different MAE variants (standard / optimized / reconstruction)
#
# This will launch several training runs sequentially and write:
#   - checkpoints  ->  ./checkpoints/<TAG>/
#   - logs (stdout) -> ./logs/<TAG>.log
#
# Dataset: Electricity
# Pred length: 96
# Seq length: 512
# Model dim: 256

set -euo pipefail

ROOT="./dataset/electricity"
DATA="electricity"
LOG_DIR="./logs"
CKPT_DIR="./checkpoints"

mkdir -p "${LOG_DIR}"
mkdir -p "${CKPT_DIR}"

COMMON_ARGS=(
  --root_path "${ROOT}"
  --data "${DATA}"
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
)

run_exp() {
  local tag="$1"
  shift
  local save_path="${CKPT_DIR}/${tag}"

  echo "============================================================"
  echo "Running experiment: ${tag}"
  echo "  save_path = ${save_path}"
  echo "  extra args = $*"
  echo "============================================================"

  python scripts/train_time_me.py \
    "${COMMON_ARGS[@]}" \
    --save_path "${save_path}" \
    "$@" \
    > "${LOG_DIR}/${tag}.log" 2>&1
}

###############################################################################
# Experiments
###############################################################################

# 1) Standard MAE, no Coupled-Mamba
run_exp "electricity_p96_mae_std_no_cm_512"

# 2) Standard MAE, with Coupled-Mamba
run_exp "electricity_p96_mae_std_cm_512" \
  --use_enhanced_fusion

# 3) Optimized MAE, no Coupled-Mamba
run_exp "electricity_p96_mae_optim_no_cm_512" \
  --use_optimized_mae

# 4) Reconstruction MAE (with feature fusion), no Coupled-Mamba
run_exp "electricity_p96_mae_recon_no_cm_512" \
  --use_reconstruction_mae \
  --use_reconstruction_features \
  --use_adaptive_norm \
  --use_global_features \
  --feature_fusion

echo "All ablation runs finished."
