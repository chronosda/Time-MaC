#!/usr/bin/env python3
"""
Evaluate a trained Time-me checkpoint under input corruption.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from configs.config import create_config_from_args, get_args
from models import TimeMEModel
from scripts.train_time_me import build_dataset_kwargs, load_best_checkpoint, set_seed, to_serializable
from utils.data_loader import MultivariateTimeSeriesDataset
from utils.input_perturb import apply_input_perturbation
from utils.metrics import calculate_metrics


def evaluate_with_corruption(model, data_loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    preds_all = []
    trues_all = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            data = apply_input_perturbation(
                data,
                noise_std=args.eval_noise_std,
                mask_ratio=args.eval_mask_ratio,
                mask_mode=args.eval_mask_mode,
                mask_block_len=args.eval_mask_block_len,
                mask_value=args.eval_mask_value,
            )
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1
            preds_all.append(output.cpu().numpy())
            trues_all.append(target.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    metrics = calculate_metrics(preds_all, trues_all)
    return total_loss / max(num_batches, 1), metrics


def main():
    args = get_args()
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint_dir is required")

    config = create_config_from_args(args)
    config.root_path = args.root_path
    config.data = args.data
    config.save_path = args.checkpoint_dir
    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()

    dataset_kwargs = build_dataset_kwargs(config)
    test_dataset = MultivariateTimeSeriesDataset(split='test', **dataset_kwargs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
    )

    model = TimeMEModel(config).to(device)
    load_best_checkpoint(model, config, device, logger=type('L', (), {'info': print, 'warning': print})())

    test_loss, test_metrics = evaluate_with_corruption(model, test_loader, criterion, device, args)
    payload = {
        'checkpoint_dir': args.checkpoint_dir,
        'dataset': args.data,
        'pred_len': config.pred_len,
        'seq_len': config.seq_len,
        'use_enhanced_fusion': bool(getattr(config, 'use_enhanced_fusion', False)),
        'eval_noise_std': args.eval_noise_std,
        'eval_mask_ratio': args.eval_mask_ratio,
        'eval_mask_mode': args.eval_mask_mode,
        'eval_mask_block_len': args.eval_mask_block_len,
        'eval_mask_value': args.eval_mask_value,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
    }

    output_path = args.output_json
    if not output_path:
        out_dir = Path(args.checkpoint_dir) / 'robustness'
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = ['clean']
        if args.eval_noise_std > 0:
            parts.append(f"noise{args.eval_noise_std:g}")
        if args.eval_mask_ratio > 0:
            parts.append(f"{args.eval_mask_mode}_mask{args.eval_mask_ratio:g}")
        output_path = str(out_dir / ('_'.join(parts) + '.json'))
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(to_serializable(payload), f, indent=2)

    print(json.dumps(to_serializable(payload), indent=2))
    print(f"Saved robustness metrics to {output_path}")


if __name__ == '__main__':
    main()
