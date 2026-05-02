#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT}/dataset"
LOG_DIR="${ROOT}/logs"
CKPT_DIR="${ROOT}/checkpoints"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

GPU="${GPU:-0}"
MAX_JOBS="${MAX_JOBS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FORCE="${FORCE:-1}"

declare -a RUN_PIDS=()
declare -a RUN_TAGS=()

is_complete_run() {
  local tag="$1"
  local metrics="${CKPT_DIR}/${tag}/results/final_metrics.json"
  local log="${CKPT_DIR}/${tag}/training.log"
  [[ -f "${metrics}" ]] && [[ -f "${log}" ]] && grep -q "Test Metrics:" "${log}"
}

dataset_root() {
  case "$1" in
    electricity) printf '%s\n' "${DATA_ROOT}/electricity" ;;
    traffic) printf '%s\n' "${DATA_ROOT}/traffic" ;;
    *) echo "Unknown large dataset: $1" >&2; return 1 ;;
  esac
}

refresh_running() {
  local next_pids=()
  local next_tags=()
  local i
  for i in "${!RUN_PIDS[@]}"; do
    if kill -0 "${RUN_PIDS[$i]}" 2>/dev/null; then
      next_pids+=("${RUN_PIDS[$i]}")
      next_tags+=("${RUN_TAGS[$i]}")
    fi
  done
  RUN_PIDS=("${next_pids[@]}")
  RUN_TAGS=("${next_tags[@]}")
}

wait_for_slot() {
  refresh_running
  while (( ${#RUN_PIDS[@]} >= MAX_JOBS )); do
    sleep 20
    refresh_running
  done
}

launch_one() {
  local dataset="$1"
  local pred_len="$2"
  local tag="${dataset}-standard-time-me-p${pred_len}"
  local root_path
  root_path="$(dataset_root "${dataset}")"

  if [[ "${FORCE}" != "1" ]] && is_complete_run "${tag}"; then
    echo "[skip] ${tag} already complete"
    return 0
  fi

  wait_for_slot
  echo "[launch] ${tag} batch_size=${BATCH_SIZE}"
  setsid -f bash -lc "
    cd '${ROOT}' && \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    exec python '${ROOT}/scripts/train_time_me.py' \
      --data '${dataset}' \
      --root_path '${root_path}' \
      --save_path '${CKPT_DIR}/${tag}' \
      --data_split custom_standard \
      --pred_len '${pred_len}' \
      --seq_len 512 \
      --d_model 512 \
      --batch_size '${BATCH_SIZE}' \
      --epochs 15 \
      --learning_rate 0.001 \
      --num_workers '${NUM_WORKERS}' \
      --use_gpu \
      --gpu '${GPU}' \
      --vlm_type clip \
      > '${LOG_DIR}/${tag}.log' 2>&1 < /dev/null
  "
  sleep 1
  local pid
  pid="$(pgrep -f "${CKPT_DIR}/${tag}" | head -n 1 || true)"
  if [[ -z "${pid}" ]]; then
    echo "[error] failed to locate process for ${tag}" >&2
    return 1
  fi
  RUN_PIDS+=("${pid}")
  RUN_TAGS+=("${tag}")
  echo "[pid] ${tag} -> ${pid}"
}

for dataset in electricity traffic; do
  for pred_len in 96 192 336 720; do
    launch_one "${dataset}" "${pred_len}"
  done
done

refresh_running
while (( ${#RUN_PIDS[@]} > 0 )); do
  echo "[wait] active=${#RUN_PIDS[@]} tags=${RUN_TAGS[*]}"
  sleep 30
  refresh_running
done

echo "[done] large benchmark reruns finished"
