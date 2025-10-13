#!/usr/bin/env python3
"""
Training script for Time-me model
Enhanced Time-VLM with Coupled-Mamba Fusion
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import time
import logging
from tqdm import tqdm

# Optional: lightweight process info without extra deps
import resource

# Add project root to path dynamically
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models import TimeMEModel
from configs.config import create_config_from_args, get_args
from utils.data_loader import MultivariateTimeSeriesDataset
from utils.conformal_plugin import ConformalCalibrator
from utils.metrics import calculate_metrics


def setup_logging(save_path):
    """Setup logging configuration"""
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

    # Track performance metrics
    epoch_start = time.time()
    num_samples = 0
    if device.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = data.shape[0]
        num_samples += batch_size
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg Loss': f'{total_loss/num_batches:.6f}'
        })

    epoch_dur = time.time() - epoch_start
    samples_per_sec = num_samples / epoch_dur if epoch_dur > 0 else 0.0
    gpu_peak_mem = None
    if device.type == 'cuda':
        try:
            gpu_peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        except Exception:
            gpu_peak_mem = None

    return total_loss / num_batches, {
        'epoch_time_sec': epoch_dur,
        'throughput_samples_per_sec': samples_per_sec,
        'gpu_peak_mem_mb': gpu_peak_mem,
    }


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # Calculate additional metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = calculate_metrics(all_predictions, all_targets)

    return total_loss / num_batches, metrics


def run_conformal_calibration(model, val_loader, device, config, logger):
    """Fit conformal calibrator on validation residuals and save calibration."""
    try:
        # Load best checkpoint if it exists
        best_path = os.path.join(config.save_path, 'checkpoint_epoch_0.pth')
        # Prefer last saved checkpoint in the directory
        try:
            ckpts = [p for p in os.listdir(config.save_path) if p.startswith('checkpoint_epoch_') and p.endswith('.pth')]
            if ckpts:
                last_ckpt = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
                best_path = os.path.join(config.save_path, last_ckpt)
        except Exception:
            pass
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=device)
            try:
                model.load_state_dict(state['model_state_dict'])
            except Exception:
                logger.warning('Could not load state dict for conformal calibration; using current weights.')
        model.eval()

        preds_list, trues_list = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds_list.append(output.detach().cpu().numpy())
                trues_list.append(target.detach().cpu().numpy())
        if not preds_list:
            logger.warning('No validation batches collected for conformal calibration.')
            return False
        preds = np.concatenate(preds_list, axis=0)  # [N, pred_len, C]
        trues = np.concatenate(trues_list, axis=0)
        # Align shapes: train_time_me uses [B, pred_len, C]
        residuals = np.abs(trues - preds)

        # Estimate scale
        scale_method = getattr(config, 'conformal_scale', 'mad')
        if scale_method == 'mad':
            med = np.median(residuals, axis=0, keepdims=True)
            s_val = np.median(np.abs(residuals - med), axis=0) / 0.6745
            s_val = np.maximum(s_val, 1e-6)
        elif scale_method == 'std':
            s_val = residuals.std(axis=0)
            s_val = np.maximum(s_val, 1e-6)
        else:  # global_mad
            med = np.median(residuals)
            s_scalar = np.median(np.abs(residuals - med)) / 0.6745
            s_scalar = max(float(s_scalar), 1e-6)
            s_val = np.full_like(residuals[0:1], s_scalar)

        max_th = getattr(config, 'conformal_max_threshold', -1.0)
        max_th = None if (max_th is None or max_th <= 0) else float(max_th)
        calib = ConformalCalibrator(
            method=getattr(config, 'conformal_method', 'crc'),
            alpha=getattr(config, 'conformal_alpha', 0.10),
            hpd_level=getattr(config, 'conformal_hpd_level', 0.95),
            num_dir=getattr(config, 'conformal_num_dir', 1000),
            delta=getattr(config, 'conformal_delta', 0.05),
        )
        calib.fit(residuals, s_val, lam_hi=max_th)
        os.makedirs('checkpoints', exist_ok=True)
        np.savez(os.path.join('checkpoints', 'conformal_calib.npz'),
                 lam_hat=calib.lam_hat_, scale=s_val,
                 method=getattr(config, 'conformal_method', 'crc'),
                 alpha=getattr(config, 'conformal_alpha', 0.10),
                 scale_method=scale_method)
        logger.info(f"[Conformal] calibrated lam_hat={calib.lam_hat_:.6f}")
        return True
    except Exception as e:
        logger.warning(f"[Conformal] calibration failed: {e}")
        return False


def apply_conformal_and_save(model, test_loader, device, logger):
    """Apply saved calibration to test predictions; save intervals and metrics."""
    calib_path = os.path.join('checkpoints', 'conformal_calib.npz')
    if not os.path.exists(calib_path):
        logger.warning('[Conformal] calibration file not found; skipping application.')
        return
    z = np.load(calib_path)
    lam_hat = float(z['lam_hat'])
    s_val = z['scale']
    exceed_sum = 0
    width_sum = 0.0
    count_sum = 0
    lower_list, upper_list, pred_list, true_list = [], [], [], []

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_np = output.detach().cpu().numpy()
            true_np = target.detach().cpu().numpy()
            s_batch = np.broadcast_to(s_val, pred_np.shape)
            lower = pred_np - lam_hat * s_batch
            upper = pred_np + lam_hat * s_batch
            exceed = ((true_np < lower) | (true_np > upper)).sum()
            width = (upper - lower).sum()
            exceed_sum += int(exceed)
            width_sum += float(width)
            count_sum += int(true_np.size)
            lower_list.append(lower.astype('float32'))
            upper_list.append(upper.astype('float32'))
            pred_list.append(pred_np.astype('float32'))
            true_list.append(true_np.astype('float32'))

    if count_sum > 0:
        miscov = exceed_sum / count_sum
        avg_width = width_sum / count_sum
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('results', 'conformal_metrics.txt'), 'a') as f:
            f.write(f"alpha={z.get('alpha', 0.10):.4f}, lam_hat={lam_hat:.6f}, miscoverage={miscov:.6f}, avg_width={avg_width:.6f}\n")
        os.makedirs('predictions', exist_ok=True)
        np.save(os.path.join('predictions', 'time_me_interval_lower.npy'), np.concatenate(lower_list, axis=0))
        np.save(os.path.join('predictions', 'time_me_interval_upper.npy'), np.concatenate(upper_list, axis=0))
        np.save(os.path.join('predictions', 'time_me_pred.npy'), np.concatenate(pred_list, axis=0))
        np.save(os.path.join('predictions', 'time_me_true.npy'), np.concatenate(true_list, axis=0))
        logger.info(f"[Conformal] applied: miscoverage={miscov:.6f}, avg_width={avg_width:.6f}")


def save_model(model, optimizer, scheduler, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'model_info': model.get_model_info()
    }

    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    logging.info(f'Model saved to {checkpoint_path}')


def main():
    """Main training function"""
    args = get_args()
    config = create_config_from_args(args)

    # Setup logging
    logger = setup_logging(args.save_path)
    logger.info("Starting Time-me training")
    logger.info(f"Configuration: {config}")

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    model = TimeMEModel(config).to(device)
    logger.info(f"Model created: {model.get_model_info()}")

    # Create data loaders (multivariate)
    train_dataset = MultivariateTimeSeriesDataset(
        root_path=args.root_path,
        data=args.data,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        split='train'
    )

    val_dataset = MultivariateTimeSeriesDataset(
        root_path=args.root_path,
        data=args.data,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        split='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True
    )

    # Update config enc_in/c_out based on dataset
    try:
        n_features = train_dataset.data.shape[1]
        config.enc_in = n_features
        config.c_out = n_features
        logger.info(f"Detected features: enc_in=c_out={n_features}")
    except Exception:
        logger.warning("Could not infer feature size; using config defaults.")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Setup training components
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15

    for epoch in range(config.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        # Training
        train_loss, perf = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f"Training Loss: {train_loss:.6f}")
        logger.info(f"Perf: time={perf['epoch_time_sec']:.2f}s, throughput={perf['throughput_samples_per_sec']:.2f} samples/s, gpu_peak_mem={(perf['gpu_peak_mem_mb'] and round(perf['gpu_peak_mem_mb'],2))}")

        # Validation
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        logger.info(f"Validation Loss: {val_loss:.6f}")
        logger.info(f"Validation Metrics: {val_metrics}")

        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, scheduler, epoch, val_loss, args.save_path)
            patience_counter = 0
            logger.info("New best model saved!")
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}/{early_stop_patience}")

        # Early stopping
        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info("Training completed!")
    # Conformal calibration + application (optional)
    if getattr(config, 'conformal_enable', False):
        ok = run_conformal_calibration(model, val_loader, device, config, logger)
        if ok:
            apply_conformal_and_save(model, val_loader, device, logger)  # can choose test_loader; using val for quick explainability


if __name__ == "__main__":
    main()
