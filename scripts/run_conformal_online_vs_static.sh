#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/dataset}"
LOG_DIR="${ROOT}/logs"
CKPT_DIR="${ROOT}/checkpoints"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

DATASET="${DATASET:-ETTm2}"
PRED_LEN="${PRED_LEN:-96}"
SEQ_LEN="${SEQ_LEN:-512}"
GPU="${GPU:-0}"
TAG_PREFIX="${TAG_PREFIX:-${DATASET,,}-conformal-p${PRED_LEN}}"
SAVE_BASE="${SAVE_BASE:-${CKPT_DIR}}"

COMMON_ARGS=(
  --data "${DATASET}"
  --root_path "${DATA_ROOT}"
  --save_path ""
  --pred_len "${PRED_LEN}"
  --seq_len "${SEQ_LEN}"
  --d_model "${D_MODEL:-512}"
  --batch_size "${BATCH_SIZE:-16}"
  --epochs "${EPOCHS:-15}"
  --learning_rate "${LEARNING_RATE:-0.001}"
  --num_workers "${NUM_WORKERS:-4}"
  --data_split "${DATA_SPLIT:-ratio}"
  --train_ratio "${TRAIN_RATIO:-0.7}"
  --val_ratio "${VAL_RATIO:-0.1}"
  --calib_ratio "${CALIB_RATIO:-0.1}"
  --test_ratio "${TEST_RATIO:-0.1}"
  --vlm_type "${VLM_TYPE:-clip}"
  --conformal_enable
  --conformal_alpha "${CONFORMAL_ALPHA:-0.10}"
  --conformal_scale "${CONFORMAL_SCALE:-mad}"
  --conformal_update_lr "${CONFORMAL_UPDATE_LR:-0.005}"
  --conformal_buffer_size "${CONFORMAL_BUFFER_SIZE:-4096}"
  --conformal_alpha_min "${CONFORMAL_ALPHA_MIN:-0.005}"
  --conformal_alpha_max "${CONFORMAL_ALPHA_MAX:-0.30}"
  --conformal_recompute_method "${CONFORMAL_RECOMPUTE_METHOD:-bq}"
  --conformal_update_block_size "${CONFORMAL_UPDATE_BLOCK_SIZE:-8}"
)

if [[ "${USE_GPU:-1}" == "1" ]]; then
  COMMON_ARGS+=(--use_gpu --gpu "${GPU}")
fi

run_one() {
  local mode="$1"
  local tag="${TAG_PREFIX}-${mode}"
  local save_path="${SAVE_BASE}/${tag}"
  local log_path="${LOG_DIR}/${tag}.log"

  echo "[Conformal Compare] starting ${tag}"
  python3 "${ROOT}/scripts/train_time_me.py" \
    "${COMMON_ARGS[@]}" \
    --save_path "${save_path}" \
    --conformal_mode "${mode}" \
    > "${log_path}" 2>&1
  echo "[Conformal Compare] finished ${tag}"
}

run_one static
run_one online

python3 "${ROOT}/scripts/summarize_conformal_compare.py" \
  "${SAVE_BASE}/${TAG_PREFIX}-static" \
  "${SAVE_BASE}/${TAG_PREFIX}-online"

echo "Conformal comparison finished."
