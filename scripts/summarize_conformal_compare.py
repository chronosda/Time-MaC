#!/usr/bin/env python3
"""Summarize static vs online conformal experiment outputs into CSV/JSON."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_metrics_txt(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    line = path.read_text(encoding="utf-8").strip().splitlines()
    if not line:
        return {}
    out: dict[str, Any] = {}
    for part in line[-1].split(","):
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def summarize_run(run_dir: Path) -> dict[str, Any]:
    results_dir = run_dir / "results"
    pred_dir = run_dir / "predictions"

    summary: dict[str, Any] = {"run_dir": str(run_dir), "tag": run_dir.name}
    summary.update(_load_json(results_dir / "final_metrics.json"))
    summary["conformal_metrics_json"] = _load_json(results_dir / "conformal_metrics.json")
    summary["conformal_metrics_txt"] = _parse_metrics_txt(results_dir / "conformal_metrics.txt")

    lambda_hist = pred_dir / "time_me_lambda_history.npy"
    alpha_hist = pred_dir / "time_me_alpha_history.npy"
    if lambda_hist.exists():
        arr = np.load(lambda_hist)
        summary["lambda_history_shape"] = list(arr.shape)
        summary["lambda_history_mean"] = float(arr.mean())
        summary["lambda_history_last_mean"] = float(arr[-1].mean())
    if alpha_hist.exists():
        arr = np.load(alpha_hist)
        summary["alpha_history_shape"] = list(arr.shape)
        summary["alpha_history_mean"] = float(arr.mean())
        summary["alpha_history_last_mean"] = float(arr[-1].mean())
    return summary


def flatten_for_csv(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "tag": summary.get("tag"),
        "run_dir": summary.get("run_dir"),
    }

    test_metrics = summary.get("test_metrics", {})
    if isinstance(test_metrics, dict):
        for key, value in test_metrics.items():
            row[f"test_{key}"] = value

    conformal_json = summary.get("conformal_metrics_json", {})
    if isinstance(conformal_json, dict):
        for key, value in conformal_json.items():
            row[f"conformal_json_{key}"] = value

    conformal_txt = summary.get("conformal_metrics_txt", {})
    if isinstance(conformal_txt, dict):
        for key, value in conformal_txt.items():
            row[f"conformal_txt_{key}"] = value

    for key in (
        "lambda_history_shape",
        "lambda_history_mean",
        "lambda_history_last_mean",
        "alpha_history_shape",
        "alpha_history_mean",
        "alpha_history_last_mean",
    ):
        if key in summary:
            row[key] = summary[key]
    return row


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: summarize_conformal_compare.py <run_dir1> <run_dir2> [<run_dir3> ...]")
        return 1

    run_dirs = [Path(p).resolve() for p in argv[1:]]
    summaries = [summarize_run(run_dir) for run_dir in run_dirs]
    rows = [flatten_for_csv(s) for s in summaries]

    out_dir = run_dirs[0] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "conformal_compare_summary.json"
    csv_path = out_dir / "conformal_compare_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote JSON summary to {json_path}")
    print(f"Wrote CSV summary to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
