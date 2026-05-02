#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT}/dataset"
LOG_DIR="${ROOT}/logs"
CKPT_DIR="${ROOT}/checkpoints"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

FORCE="${FORCE:-0}"
GPU="${GPU:-0}"

is_complete_run() {
  local tag="$1"
  local metrics="${CKPT_DIR}/${tag}/results/final_metrics.json"
  local log="${CKPT_DIR}/${tag}/training.log"
  [[ -f "${metrics}" ]] && [[ -f "${log}" ]] && grep -q "Test Metrics:" "${log}"
}

dataset_root() {
  local dataset="$1"
  case "${dataset}" in
    ETTm1|ETTm2|ETTh1|ETTh2)
      printf '%s\n' "${DATA_ROOT}"
      ;;
    electricity)
      printf '%s\n' "${DATA_ROOT}/electricity"
      ;;
    weather)
      printf '%s\n' "${DATA_ROOT}/weather"
      ;;
    traffic)
      printf '%s\n' "${DATA_ROOT}/traffic"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

dataset_split() {
  local dataset="$1"
  case "${dataset}" in
    ETTm1|ETTm2|ETTh1|ETTh2)
      printf '%s\n' "ett_standard"
      ;;
    electricity|weather|traffic)
      printf '%s\n' "custom_standard"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local pred_len="$2"
  local tag="$3"
  local split
  local root_path

  split="$(dataset_split "${dataset}")"
  root_path="$(dataset_root "${dataset}")"

  if [[ "${FORCE}" != "1" ]] && is_complete_run "${tag}"; then
    echo "[skip] ${tag} already has final metrics and test output"
    return 0
  fi

  echo "[run] ${tag} dataset=${dataset} pred_len=${pred_len} split=${split}"
  python "${ROOT}/scripts/train_time_me.py" \
    --data "${dataset}" \
    --root_path "${root_path}" \
    --save_path "${CKPT_DIR}/${tag}" \
    --data_split "${split}" \
    --pred_len "${pred_len}" \
    --seq_len 512 \
    --d_model 512 \
    --batch_size 16 \
    --epochs 15 \
    --learning_rate 0.001 \
    --num_workers 4 \
    --use_gpu \
    --gpu "${GPU}" \
    --vlm_type clip \
    > "${LOG_DIR}/${tag}.log" 2>&1
}

run_dataset() {
  local dataset="$1"
  local prefix="$2"
  run_one "${dataset}" 96  "${prefix}-p96"
  run_one "${dataset}" 192 "${prefix}-p192"
  run_one "${dataset}" 336 "${prefix}-p336"
  run_one "${dataset}" 720 "${prefix}-p720"
}

run_dataset "ETTm1" "ettm1-standard-time-me"
run_dataset "ETTm2" "ettm2-standard-time-me"
run_dataset "ETTh1" "etth1-standard-time-me"
run_dataset "ETTh2" "etth2-standard-time-me"
run_dataset "electricity" "electricity-standard-time-me"
run_dataset "weather" "weather-standard-time-me"
run_dataset "traffic" "traffic-standard-time-me"
