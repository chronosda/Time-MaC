#!/usr/bin/env python3
"""
Monitor one or more training log files, wait for completion, and write summaries.

Usage:
  python scripts/monitor_and_summarize.py logs/file1.log [logs/file2.log ...]

Outputs per-log summaries to logs/summary_<basename>.txt and a combined summary
to logs/combined_summary.txt in the same directory as the first log.
"""

import sys
import time
import re
import os
import ast
from typing import Dict, Any, List, Tuple


def parse_log(content: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "completed": "Training completed!" in content,
        "last_training_loss": None,
        "last_validation_loss": None,
        "last_validation_metrics": None,
        "best_checkpoint": None,
        "early_stopped_epoch": None,
    }

    tr_losses = re.findall(r"Training Loss:\s*([0-9.]+)", content)
    if tr_losses:
        try:
            result["last_training_loss"] = float(tr_losses[-1])
        except Exception:
            pass

    val_losses = re.findall(r"Validation Loss:\s*([0-9.]+)", content)
    if val_losses:
        try:
            result["last_validation_loss"] = float(val_losses[-1])
        except Exception:
            pass

    # Validation Metrics dict on one line
    metrics_matches = re.findall(r"Validation Metrics:\s*(\{.*\})", content)
    if metrics_matches:
        try:
            result["last_validation_metrics"] = ast.literal_eval(metrics_matches[-1])
        except Exception:
            pass

    # Last saved checkpoint
    ckpt_matches = re.findall(r"Model saved to\s+(.+)", content)
    if ckpt_matches:
        result["best_checkpoint"] = ckpt_matches[-1].strip()

    # Early stopping epoch
    es = re.findall(r"Early stopping triggered at epoch\s+(\d+)", content)
    if es:
        result["early_stopped_epoch"] = int(es[-1])

    return result


def wait_for_completion(paths: List[str], poll_seconds: int = 60) -> Dict[str, Dict[str, Any]]:
    """Poll the given logs until each contains 'Training completed!'"""
    statuses: Dict[str, Dict[str, Any]] = {p: {} for p in paths}
    pending = set(paths)
    while pending:
        for p in list(pending):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except FileNotFoundError:
                continue
            info = parse_log(content)
            statuses[p] = info
            if info.get("completed"):
                pending.remove(p)
        if pending:
            time.sleep(poll_seconds)
    return statuses


def write_summaries(statuses: Dict[str, Dict[str, Any]]) -> str:
    """Write per-log summaries and a combined summary; return combined path."""
    if not statuses:
        return ""
    # Combined summary path (in logs/ next to first log)
    first_log = list(statuses.keys())[0]
    base_dir = os.path.dirname(first_log) or "."
    combined_path = os.path.join(base_dir, "combined_summary.txt")

    lines_combined: List[str] = ["Training summaries (auto-generated):\n"]
    for log_path, info in statuses.items():
        base = os.path.basename(log_path)
        summary_path = os.path.join(base_dir, f"summary_{base}.txt")
        lines = [
            f"Log: {log_path}",
            f"Completed: {info.get('completed')}",
            f"EarlyStoppedEpoch: {info.get('early_stopped_epoch')}",
            f"LastTrainingLoss: {info.get('last_training_loss')}",
            f"LastValidationLoss: {info.get('last_validation_loss')}",
            f"LastValidationMetrics: {info.get('last_validation_metrics')}",
            f"BestCheckpoint: {info.get('best_checkpoint')}",
            "",
        ]
        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write("\n".join(lines))
        lines_combined.extend(lines)

    with open(combined_path, "w", encoding="utf-8") as cf:
        cf.write("\n".join(lines_combined))
    return combined_path


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: monitor_and_summarize.py <log1> [<log2> ...]")
        return 1
    logs = argv[1:]
    print("Monitoring logs:", ", ".join(logs))
    statuses = wait_for_completion(logs, poll_seconds=60)
    combined = write_summaries(statuses)
    print("Summaries written.")
    if combined:
        print("Combined summary:", combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

