#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT}/dataset"
LOG_DIR="${ROOT}/logs"
CKPT_DIR="${ROOT}/checkpoints"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

FORCE="${FORCE:-0}"
GPU="${GPU:-0}"
MAX_JOBS="${MAX_JOBS:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"

declare -a RUN_PIDS=()
declare -a RUN_TAGS=()

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
    sleep 15
    refresh_running
  done
}

launch_one() {
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

  wait_for_slot
  echo "[launch] ${tag} dataset=${dataset} pred_len=${pred_len} split=${split}"
  setsid -f bash -lc "
    exec python '${ROOT}/scripts/train_time_me.py' \
      --data '${dataset}' \
      --root_path '${root_path}' \
      --save_path '${CKPT_DIR}/${tag}' \
      --data_split '${split}' \
      --pred_len '${pred_len}' \
      --seq_len 512 \
      --d_model 512 \
      --batch_size 16 \
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

launch_dataset() {
  local dataset="$1"
  local prefix="$2"
  launch_one "${dataset}" 96  "${prefix}-p96"
  launch_one "${dataset}" 192 "${prefix}-p192"
  launch_one "${dataset}" 336 "${prefix}-p336"
  launch_one "${dataset}" 720 "${prefix}-p720"
}

launch_dataset "ETTm1" "ettm1-standard-time-me"
launch_dataset "ETTm2" "ettm2-standard-time-me"
launch_dataset "ETTh1" "etth1-standard-time-me"
launch_dataset "ETTh2" "etth2-standard-time-me"
launch_dataset "electricity" "electricity-standard-time-me"
launch_dataset "weather" "weather-standard-time-me"
launch_dataset "traffic" "traffic-standard-time-me"

refresh_running
while (( ${#RUN_PIDS[@]} > 0 )); do
  echo "[wait] active=${#RUN_PIDS[@]} tags=${RUN_TAGS[*]}"
  sleep 20
  refresh_running
done

echo "[done] all scheduled benchmark runs finished"
