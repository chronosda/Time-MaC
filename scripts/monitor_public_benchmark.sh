#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${ROOT}/checkpoints"

printf '== active processes ==\n'
ps -ef | rg '/home/chronos/Time-me/scripts/train_time_me.py|run_public_benchmark_parallel' || true

printf '\n== gpu ==\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
printf '%s\n' '---'
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true

printf '\n== completed runs ==\n'
python - <<'PY'
import pathlib
for d in sorted(pathlib.Path("checkpoints").glob("*-standard-time-me-p*")):
    fm = d / "results" / "final_metrics.json"
    log = d / "training.log"
    done = fm.exists() and log.exists() and ("Test Metrics:" in log.read_text(errors="ignore"))
    print(f"{d.name}\t{'done' if done else 'pending'}")
PY
