#!/usr/bin/env python3
"""Plot publication-style figures for missingness robustness ablations."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/home/chronos/Time-me/docs/figures")


RANDOM_MASK = {
    "ratio": np.array([0.2, 0.3, 0.4, 0.5]),
    "mse_no_cm": np.array([0.21654, 0.26781, 0.33105, 0.40632]),
    "mse_cm": np.array([0.21125, 0.25771, 0.31652, 0.38791]),
    "mae_no_cm": np.array([0.33243, 0.38071, 0.43259, 0.48721]),
    "mae_cm": np.array([0.32425, 0.36916, 0.41903, 0.47265]),
}

BLOCK_MASK = {
    "ratio": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    "mse_no_cm": np.array([0.17865, 0.21275, 0.25119, 0.29076, 0.32823]),
    "mse_cm": np.array([0.18032, 0.21073, 0.24642, 0.28314, 0.31925]),
    "mae_no_cm": np.array([0.28865, 0.32435, 0.36020, 0.39434, 0.42386]),
    "mae_cm": np.array([0.28800, 0.32018, 0.35359, 0.38575, 0.41464]),
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#404040",
            "axes.linewidth": 0.9,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#D7DCE2",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
        }
    )


def style_axis(ax: plt.Axes, x: np.ndarray) -> None:
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim(x.min() - 0.02, x.max() + 0.02)
    ax.set_xticks(x)
    ax.set_xlabel("Mask ratio")


def add_series(ax: plt.Axes, x: np.ndarray, y_no_cm: np.ndarray, y_cm: np.ndarray, ylabel: str) -> None:
    color_baseline = "#8C97A5"
    color_ours = "#1F6F78"
    accent = "#D97757"

    ax.plot(
        x,
        y_no_cm,
        label="no-CM",
        color=color_baseline,
        linewidth=2.0,
        marker="o",
        markersize=5.4,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linestyle=(0, (4, 2)),
    )
    ax.plot(
        x,
        y_cm,
        label="CM (Ours)",
        color=color_ours,
        linewidth=2.35,
        marker="D",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.1,
    )

    ax.fill_between(x, y_cm, y_no_cm, where=y_no_cm >= y_cm, color=color_ours, alpha=0.08)
    ax.fill_between(x, y_cm, y_no_cm, where=y_no_cm < y_cm, color=accent, alpha=0.06)
    ax.set_ylabel(ylabel)


def annotate_gap(ax: plt.Axes, x: float, y0: float, y1: float, text: str) -> None:
    y_low, y_high = sorted((y0, y1))
    ax.vlines(x, y_low, y_high, color="#D97757", linewidth=1.4, alpha=0.9)
    ax.text(
        x + 0.008,
        (y0 + y1) / 2.0,
        text,
        fontsize=8.8,
        color="#9E4F39",
        va="center",
        ha="left",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.92},
    )


def format_y_ticks(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))


def plot_main_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    add_series(
        axes[0],
        RANDOM_MASK["ratio"],
        RANDOM_MASK["mse_no_cm"],
        RANDOM_MASK["mse_cm"],
        ylabel="MSE",
    )
    style_axis(axes[0], RANDOM_MASK["ratio"])
    axes[0].set_title("Random masking")
    format_y_ticks(axes[0])
    random_gain = (RANDOM_MASK["mse_no_cm"][-1] - RANDOM_MASK["mse_cm"][-1]) / RANDOM_MASK["mse_no_cm"][-1] * 100
    annotate_gap(
        axes[0],
        RANDOM_MASK["ratio"][-1],
        RANDOM_MASK["mse_no_cm"][-1],
        RANDOM_MASK["mse_cm"][-1],
        f"-{random_gain:.1f}%",
    )

    add_series(
        axes[1],
        BLOCK_MASK["ratio"],
        BLOCK_MASK["mse_no_cm"],
        BLOCK_MASK["mse_cm"],
        ylabel="MSE",
    )
    style_axis(axes[1], BLOCK_MASK["ratio"])
    axes[1].set_title("Block masking")
    format_y_ticks(axes[1])
    block_gain = (BLOCK_MASK["mse_no_cm"][-1] - BLOCK_MASK["mse_cm"][-1]) / BLOCK_MASK["mse_no_cm"][-1] * 100
    annotate_gap(
        axes[1],
        BLOCK_MASK["ratio"][-1],
        BLOCK_MASK["mse_no_cm"][-1],
        BLOCK_MASK["mse_cm"][-1],
        f"-{block_gain:.1f}%",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "missingness_ablation_main.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "missingness_ablation_main.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "missingness_ablation_main.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_full_metrics_figure() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)

    panels = [
        (axes[0, 0], RANDOM_MASK["ratio"], RANDOM_MASK["mse_no_cm"], RANDOM_MASK["mse_cm"], "Random masking", "MSE"),
        (axes[0, 1], BLOCK_MASK["ratio"], BLOCK_MASK["mse_no_cm"], BLOCK_MASK["mse_cm"], "Block masking", "MSE"),
        (axes[1, 0], RANDOM_MASK["ratio"], RANDOM_MASK["mae_no_cm"], RANDOM_MASK["mae_cm"], "", "MAE"),
        (axes[1, 1], BLOCK_MASK["ratio"], BLOCK_MASK["mae_no_cm"], BLOCK_MASK["mae_cm"], "", "MAE"),
    ]

    for ax, x, y_no_cm, y_cm, title, ylabel in panels:
        add_series(ax, x, y_no_cm, y_cm, ylabel=ylabel)
        style_axis(ax, x)
        if title:
            ax.set_title(title)
        format_y_ticks(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "missingness_ablation_full_metrics.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "missingness_ablation_full_metrics.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "missingness_ablation_full_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    plot_main_figure()
    plot_full_metrics_figure()


if __name__ == "__main__":
    main()
