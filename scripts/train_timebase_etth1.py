#!/usr/bin/env python3
"""
Train TimeBase (plug-and-play) forecaster on ETTh1 inside the Time-me project.

This script reuses Time-me's data pipeline and metrics, but swaps in the
TimeBase model as a drop-in forecaster for fair comparison.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Ensure project root is on sys.path, consistent with other scripts
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from utils.data_loader import MultivariateTimeSeriesDataset
from utils.metrics import calculate_metrics
from models.timebase_adapter import build_timebase_for_dataset


def setup_logging(save_path: str) -> logging.Logger:
    """Setup file + console logging."""
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, "training_timebase_etth1.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("TimeBaseETTh1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TimeBase forecaster on ETTh1 (Time-me integration)."
    )

    # Data
    parser.add_argument(
        "--root_path",
        type=str,
        default="./dataset",
        help="Root path of dataset directory.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ETTh1",
        help="Dataset name (CSV without extension).",
    )
    parser.add_argument("--seq_len", type=int, default=720, help="Input sequence length.")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="Prediction sequence length."
    )

    # TimeBase hyperparameters
    parser.add_argument(
        "--period_len",
        type=int,
        default=24,
        help="Seasonal period length (e.g., 24 for hourly ETTh1).",
    )
    parser.add_argument(
        "--basis_num",
        type=int,
        default=6,
        help="Number of basis functions (R in the paper).",
    )
    parser.add_argument(
        "--lambda_orth",
        type=float,
        default=0.0,
        help="Weight for orthogonal regularization term.",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer.",
    )

    # Device and IO
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use CUDA if available."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints/timebase_etth1",
        help="Directory to save checkpoints and logs.",
    )

    return parser.parse_args()


def build_dataloaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    train_dataset = MultivariateTimeSeriesDataset(
        root_path=args.root_path,
        data=args.data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split="train",
    )
    val_dataset = MultivariateTimeSeriesDataset(
        root_path=args.root_path,
        data=args.data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split="val",
    )
    test_dataset = MultivariateTimeSeriesDataset(
        root_path=args.root_path,
        data=args.data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split="test",
    )

    n_features = train_dataset.data.shape[1]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, n_features


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        out = model(data)

        if isinstance(out, tuple):
            preds, orth_term = out
            loss = criterion(preds, target) + orth_term
        else:
            preds = out
            loss = criterion(preds, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(num_batches, 1)


def eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    preds_list, trues_list = [], []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)

            if isinstance(out, tuple):
                preds, _ = out
            else:
                preds = out

            loss = criterion(preds, target)
            total_loss += float(loss.item())
            num_batches += 1

            preds_list.append(preds.cpu().numpy())
            trues_list.append(target.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    metrics = calculate_metrics(preds, trues)
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, metrics


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_loss: float,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"timebase_etth1_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        },
        ckpt_path,
    )


def main() -> None:
    args = parse_args()

    # Resolve and create save directory
    save_dir = Path(args.save_path).expanduser().resolve()
    logger = setup_logging(str(save_dir))
    logger.info(f"Args: {args}")

    device = torch.device(
        f"cuda:{args.gpu}"
        if args.use_gpu and torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, n_features = build_dataloaders(args)
    logger.info(
        f"Loaded ETTh1 with enc_in=c_out={n_features}, "
        f"train/val/test sizes = {len(train_loader.dataset)}/"
        f"{len(val_loader.dataset)}/{len(test_loader.dataset)}"
    )

    # Model
    model = build_timebase_for_dataset(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=n_features,
        period_len=args.period_len,
        basis_num=args.basis_num,
        lambda_orth=args.lambda_orth,
    ).to(device)
    logger.info(
        f"TimeBase model created: seq_len={args.seq_len}, "
        f"pred_len={args.pred_len}, enc_in={n_features}, "
        f"period_len={args.period_len}, basis_num={args.basis_num}, "
        f"lambda_orth={args.lambda_orth}"
    )

    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_model(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Val   Loss: {val_loss:.6f}")
        logger.info(f"Val   Metrics: {val_metrics}")
        logger.info(f"LR: {optimizer.param_groups[0]['lr']:.6e}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, str(save_dir))
            logger.info("New best checkpoint saved.")

    logger.info(
        f"Training finished. Best val loss {best_val:.6f} at epoch {best_epoch}."
    )

    # Final evaluation on test set (using last weights; reload best if desired)
    test_loss, test_metrics = eval_model(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test Metrics: {test_metrics}")

    # Persist comparison-friendly metrics
    results_dir = Path("./results").expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "timebase_etth1_metrics.txt"
    with out_file.open("a") as f:
        f.write(
            f"TimeBase_ETTh1 seq_len={args.seq_len} pred_len={args.pred_len} "
            f"period_len={args.period_len} basis_num={args.basis_num} "
            f"lambda_orth={args.lambda_orth}\n"
        )
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
    logger.info(f"Saved test metrics to {out_file}")


if __name__ == "__main__":
    main()
