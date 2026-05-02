#!/usr/bin/env python3
"""
Visualize conformal prediction intervals for Time-me predictions.

Loads saved arrays from predictions/: time_me_pred.npy, time_me_true.npy,
time_me_interval_lower.npy, time_me_interval_upper.npy and plots a
single sample and variable with shaded intervals.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_arrays(pred_dir: str):
    pred = np.load(os.path.join(pred_dir, 'time_me_pred.npy'))
    true = np.load(os.path.join(pred_dir, 'time_me_true.npy'))
    lower = np.load(os.path.join(pred_dir, 'time_me_interval_lower.npy'))
    upper = np.load(os.path.join(pred_dir, 'time_me_interval_upper.npy'))
    return pred, true, lower, upper


def plot_sample(pred, true, lower, upper, index=0, channel=0, out_file=None, title=None):
    """Plot a single sample (index) and variable (channel). Shapes: [N, C, H]."""
    assert pred.ndim == 3, f"expected pred shape [N, C, H], got {pred.shape}"
    N, C, H = pred.shape
    if index < 0 or index >= N:
        raise IndexError(f"index out of range: {index} / {N}")
    if channel < 0 or channel >= C:
        raise IndexError(f"channel out of range: {channel} / {C}")

    x = np.arange(H)
    y_pred = pred[index, channel]
    y_true = true[index, channel]
    y_lo = lower[index, channel]
    y_hi = upper[index, channel]

    plt.figure(figsize=(8, 4))
    plt.fill_between(x, y_lo, y_hi, color='C0', alpha=0.2, label='interval')
    plt.plot(x, y_pred, color='C0', lw=2, label='prediction')
    plt.plot(x, y_true, color='C3', lw=1.5, ls='--', label='truth')
    plt.xlabel('horizon')
    plt.ylabel('value (normalized)')
    if title:
        plt.title(title)
    else:
        plt.title(f'Sample {index}, Channel {channel}')
    plt.legend()
    plt.tight_layout()
    if out_file:
        os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
        plt.savefig(out_file, dpi=150)
        print(f'saved figure to {out_file}')
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', type=str, default='predictions', help='directory with saved arrays')
    ap.add_argument('--index', type=int, default=0, help='sample index to plot')
    ap.add_argument('--channel', type=int, default=0, help='variable/channel index to plot')
    ap.add_argument('--out', type=str, default='', help='optional output image path')
    args = ap.parse_args()

    pred, true, lower, upper = load_arrays(args.pred_dir)
    out_file = args.out if args.out else None
    plot_sample(pred, true, lower, upper, index=args.index, channel=args.channel, out_file=out_file)


if __name__ == '__main__':
    main()

