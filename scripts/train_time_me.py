#!/usr/bin/env python3
"""
Training script for the Time-MaC/Time-me model with Coupled-Mamba fusion.
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import time
import logging
from tqdm import tqdm

# Add project root to path dynamically
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models import TimeMEModel
from configs.config import create_config_from_args, get_args
from utils.data_loader import MultivariateTimeSeriesDataset
from utils.conformal_plugin import (
    AdaptiveOnlineConformalCalibrator,
    ConformalCalibrator,
    estimate_scale_from_residuals,
)
from utils.metrics import StreamingMetricsAccumulator


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


def evaluate_epoch(model, data_loader, criterion, device):
    """Evaluate on one split."""
    model.eval()
    total_loss = 0
    num_batches = 0
    metrics_accumulator = StreamingMetricsAccumulator()

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

            metrics_accumulator.update(output.cpu().numpy(), target.cpu().numpy())

    metrics = metrics_accumulator.compute()

    return total_loss / num_batches, metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_kwargs(config):
    return {
        'root_path': config.root_path,
        'data': config.data,
        'seq_len': config.seq_len,
        'pred_len': config.pred_len,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'calib_ratio': config.calib_ratio,
        'test_ratio': config.test_ratio,
        'data_split': getattr(config, 'data_split', 'ratio'),
    }


def maybe_subset_train_dataset(dataset, config, logger):
    ratio = float(getattr(config, 'train_subset_ratio', 1.0))
    if ratio >= 1.0:
        return dataset
    if ratio <= 0.0:
        raise ValueError(f"train_subset_ratio must be in (0, 1], got {ratio}")

    total = len(dataset)
    keep = max(1, int(round(total * ratio)))
    rng_seed = int(getattr(config, 'train_subset_seed', getattr(config, 'seed', 0)))
    rng = np.random.default_rng(rng_seed)
    indices = np.sort(rng.choice(total, size=keep, replace=False))
    logger.info(f"Applying train subset ratio={ratio:.3f}: keeping {keep}/{total} train windows")
    return Subset(dataset, indices.tolist())


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def load_best_checkpoint(model, config, device, logger):
    ckpts = [
        p for p in os.listdir(config.save_path)
        if p.startswith('checkpoint_epoch_') and p.endswith('.pth')
    ]
    if not ckpts:
        logger.warning('No checkpoint found under save_path; using current in-memory weights.')
        return False

    def score_key(name):
        path = os.path.join(config.save_path, name)
        try:
            state = torch.load(path, map_location='cpu')
            return float(state.get('loss', float('inf')))
        except Exception:
            return float('inf')

    best_ckpt = min(ckpts, key=score_key)
    state = torch.load(os.path.join(config.save_path, best_ckpt), map_location=device)
    model.load_state_dict(state['model_state_dict'])
    logger.info(f'Loaded best checkpoint: {best_ckpt}')
    return True


def run_conformal_calibration(model, calib_loader, device, config, logger):
    """Fit conformal calibrator on calibration residuals and save calibration."""
    try:
        load_best_checkpoint(model, config, device, logger)
        model.eval()

        preds_list, trues_list = [], []
        with torch.no_grad():
            for data, target in calib_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds_list.append(output.detach().cpu().numpy())
                trues_list.append(target.detach().cpu().numpy())
        if not preds_list:
            logger.warning('No calibration batches collected for conformal calibration.')
            return False
        preds = np.concatenate(preds_list, axis=0)  # [N, pred_len, C]
        trues = np.concatenate(trues_list, axis=0)
        # Align shapes: train_time_me uses [B, pred_len, C]
        residuals = np.abs(trues - preds)

        scale_method = getattr(config, 'conformal_scale', 'mad')
        s_val = estimate_scale_from_residuals(residuals, scale_method)
        calib_path = os.path.join(config.save_path, 'conformal_calib.npz')
        mode = getattr(config, 'conformal_mode', 'online')
        save_payload = {
            'scale': s_val,
            'method': getattr(config, 'conformal_method', 'crc'),
            'alpha': getattr(config, 'conformal_alpha', 0.10),
            'scale_method': scale_method,
            'split': 'calib',
        }
        if mode == 'online':
            calib = AdaptiveOnlineConformalCalibrator(
                alpha=getattr(config, 'conformal_alpha', 0.10),
                update_lr=getattr(config, 'conformal_update_lr', 0.005),
                buffer_size=getattr(config, 'conformal_buffer_size', 4096),
                alpha_min=getattr(config, 'conformal_alpha_min', 0.005),
                alpha_max=getattr(config, 'conformal_alpha_max', 0.30),
                pred_len=getattr(config, 'pred_len', None),
                threshold_method=getattr(config, 'conformal_method', 'hpd'),
                recompute_method=getattr(config, 'conformal_recompute_method', 'bq'),
                hpd_level=getattr(config, 'conformal_hpd_level', 0.95),
                num_dir=getattr(config, 'conformal_num_dir', 1000),
                delta=getattr(config, 'conformal_delta', 0.05),
                max_threshold=None
                if getattr(config, 'conformal_max_threshold', -1.0) <= 0
                else float(getattr(config, 'conformal_max_threshold', -1.0)),
                update_block_size=getattr(config, 'conformal_update_block_size', 8),
                rng_seed=getattr(config, 'seed', 0),
            )
            calib.fit(residuals, s_val)
            save_payload.update(calib.state_dict(s_val))
            np.savez(calib_path, **save_payload)
            logger.info(
                "[Conformal] initialized online calibration on calib split, "
                f"mean_lambda={float(np.mean(calib.lambda_t_)):.6f}"
            )
        else:
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
            save_payload.update({'mode': np.array('static'), 'lam_hat': calib.lam_hat_})
            np.savez(calib_path, **save_payload)
            logger.info(f"[Conformal] calibrated on calib split, lam_hat={calib.lam_hat_:.6f}")
        return True
    except Exception as e:
        logger.warning(f"[Conformal] calibration failed: {e}")
        return False


def apply_conformal_and_save(model, test_loader, device, config, logger):
    """Apply saved calibration to test predictions; save intervals and metrics."""
    calib_path = os.path.join(config.save_path, 'conformal_calib.npz')
    if not os.path.exists(calib_path):
        logger.warning('[Conformal] calibration file not found; skipping application.')
        return None
    z = np.load(calib_path)
    saved_mode = str(z['mode']) if 'mode' in z.files else 'static'
    lam_hat = float(z['lam_hat']) if saved_mode != 'online' else None
    s_val = z['scale']
    online_calib = AdaptiveOnlineConformalCalibrator.from_npz(z) if saved_mode == 'online' else None
    exceed_sum = 0
    width_sum = 0.0
    count_sum = 0
    lower_list, upper_list, pred_list, true_list = [], [], [], []
    lambda_hist, alpha_hist = [], []

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_np = output.detach().cpu().numpy()
            true_np = target.detach().cpu().numpy()
            s_batch = np.broadcast_to(s_val, pred_np.shape)
            if saved_mode == 'online' and online_calib is not None:
                lower, upper = online_calib.apply(pred_np, s_batch)
                online_calib.update(pred_np, true_np, s_batch)
                lambda_hist.append(online_calib.lambda_t_.astype('float32'))
                alpha_hist.append(online_calib.alpha_t_.astype('float32'))
            else:
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
        results_dir = os.path.join(config.save_path, 'results')
        pred_dir = os.path.join(config.save_path, 'predictions')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'conformal_metrics.txt'), 'a') as f:
            if saved_mode == 'online' and online_calib is not None:
                f.write(
                    f"mode=online, alpha={float(z['alpha']):.4f}, "
                    f"final_mean_alpha={float(np.mean(online_calib.alpha_t_)):.6f}, "
                    f"final_mean_lambda={float(np.mean(online_calib.lambda_t_)):.6f}, "
                    f"miscoverage={miscov:.6f}, avg_width={avg_width:.6f}\n"
                )
            else:
                f.write(
                    f"mode=static, alpha={float(z['alpha']):.4f}, lam_hat={lam_hat:.6f}, "
                    f"miscoverage={miscov:.6f}, avg_width={avg_width:.6f}\n"
                )
        np.save(os.path.join(pred_dir, 'time_me_interval_lower.npy'), np.concatenate(lower_list, axis=0))
        np.save(os.path.join(pred_dir, 'time_me_interval_upper.npy'), np.concatenate(upper_list, axis=0))
        np.save(os.path.join(pred_dir, 'time_me_pred.npy'), np.concatenate(pred_list, axis=0))
        np.save(os.path.join(pred_dir, 'time_me_true.npy'), np.concatenate(true_list, axis=0))
        if saved_mode == 'online' and lambda_hist:
            np.save(os.path.join(pred_dir, 'time_me_lambda_history.npy'), np.stack(lambda_hist, axis=0))
        if saved_mode == 'online' and alpha_hist:
            np.save(os.path.join(pred_dir, 'time_me_alpha_history.npy'), np.stack(alpha_hist, axis=0))
        logger.info(f"[Conformal] applied: mode={saved_mode}, miscoverage={miscov:.6f}, avg_width={avg_width:.6f}")
        if saved_mode == 'online' and online_calib is not None:
            return {
                'miscoverage': miscov,
                'avg_width': avg_width,
                'final_mean_lambda': float(np.mean(online_calib.lambda_t_)),
                'final_mean_alpha': float(np.mean(online_calib.alpha_t_)),
                'mode': saved_mode,
            }
        return {'miscoverage': miscov, 'avg_width': avg_width, 'lam_hat': lam_hat, 'mode': saved_mode}
    return None


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
    config.root_path = args.root_path
    config.data = args.data
    config.save_path = args.save_path
    set_seed(config.seed)

    # Setup logging
    logger = setup_logging(args.save_path)
    logger.info("Starting Time-me training")
    logger.info(f"Configuration: {config}")

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create data loaders (multivariate)
    dataset_kwargs = build_dataset_kwargs(config)
    train_dataset = MultivariateTimeSeriesDataset(
        split='train',
        **dataset_kwargs,
    )
    raw_train_size = len(train_dataset)
    train_dataset = maybe_subset_train_dataset(train_dataset, config, logger)

    val_dataset = MultivariateTimeSeriesDataset(
        split='val',
        **dataset_kwargs,
    )

    calib_dataset = MultivariateTimeSeriesDataset(
        split='calib',
        **dataset_kwargs,
    )

    test_dataset = MultivariateTimeSeriesDataset(
        split='test',
        **dataset_kwargs,
    )

    # Update config enc_in/c_out based on dataset before model init
    try:
        n_features = train_dataset.data.shape[1]
        config.enc_in = n_features
        config.c_out = n_features
        logger.info(f"Detected features: enc_in=c_out={n_features}")
    except Exception:
        logger.warning("Could not infer feature size; using config defaults.")

    # Create model
    model = TimeMEModel(config).to(device)
    logger.info(f"Model created: {model.get_model_info()}")

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

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)} (raw={raw_train_size})")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Calibration dataset size: {len(calib_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

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
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
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
    load_best_checkpoint(model, config, device, logger)
    test_loss, test_metrics = evaluate_epoch(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test Metrics: {test_metrics}")

    metrics_path = os.path.join(config.save_path, 'results')
    os.makedirs(metrics_path, exist_ok=True)
    with open(os.path.join(metrics_path, 'final_metrics.json'), 'w') as f:
        json.dump(
            to_serializable({
                'validation_best_loss': best_val_loss,
                'test_loss': test_loss,
                'test_metrics': test_metrics,
            }),
            f,
            indent=2,
        )

    # Conformal calibration + application (optional)
    if getattr(config, 'conformal_enable', False):
        ok = run_conformal_calibration(model, calib_loader, device, config, logger)
        if ok:
            conformal_metrics = apply_conformal_and_save(model, test_loader, device, config, logger)
            if conformal_metrics is not None:
                with open(os.path.join(metrics_path, 'conformal_metrics.json'), 'w') as f:
                    json.dump(to_serializable(conformal_metrics), f, indent=2)


if __name__ == "__main__":
    main()
