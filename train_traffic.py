#!/usr/bin/env python3
"""
Time-me模型训练脚本
在traffic数据集上训练并验证coupled-mamba融合模块的有效性
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from datetime import datetime
from pathlib import Path
import json

# Add project root to path dynamically
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.coupled_mamba_fusion import CoupledMambaFusion
from configs.config import TimeMEConfig, get_args, create_config_from_args
from utils.conformal_plugin import (
    AdaptiveOnlineConformalCalibrator,
    ConformalCalibrator,
    estimate_scale_from_residuals,
)


def _split_boundaries(total_len, train_ratio, val_ratio, calib_ratio, test_ratio):
    ratios = np.array([train_ratio, val_ratio, calib_ratio, test_ratio], dtype=np.float64)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios.sum():.6f}")
    raw_lengths = ratios * total_len
    lengths = np.floor(raw_lengths).astype(int)
    for idx in np.argsort(-(raw_lengths - lengths))[: total_len - int(lengths.sum())]:
        lengths[idx] += 1
    train_len, val_len, calib_len, test_len = lengths.tolist()
    train_end = train_len
    val_end = train_end + val_len
    calib_end = val_end + calib_len
    return {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'calib': (val_end, calib_end),
        'test': (calib_end, calib_end + test_len),
    }


class TrafficDataset(Dataset):
    """Traffic数据集加载器"""
    def __init__(
        self,
        data_path,
        seq_len=96,
        pred_len=12,
        split='train',
        restrict_vars: int = -1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        calib_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 加载数据
        print(f"Loading traffic data from {data_path}...")
        df = pd.read_csv(data_path)

        # 解析日期列
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 提取数值列（排除日期列）
        data_cols = [col for col in df.columns if col != 'date']
        data = df[data_cols].values

        # optional variable restriction
        if restrict_vars is not None and restrict_vars > 0 and restrict_vars < data.shape[1]:
            data = data[:, :restrict_vars]
            print(f"Restricting variables to first {restrict_vars} columns (n_vars now={data.shape[1]})")

        boundaries = _split_boundaries(
            len(data), train_ratio, val_ratio, calib_ratio, test_ratio
        )
        train_start, train_end = boundaries['train']

        # 标准化数据（仅用训练集估计，避免泄漏）
        self.mean = np.mean(data[train_start:train_end], axis=0)
        self.std = np.std(data[train_start:train_end], axis=0) + 1e-8
        data = (data - self.mean) / self.std

        if split not in boundaries:
            raise ValueError(f"Unknown split: {split}")
        split_start, split_end = boundaries[split]
        self.data = data[split_start:split_end]

        print(f"{split} set size: {len(self.data)} samples")

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 获取输入序列和目标序列
        x = self.data[idx:idx + self.seq_len]  # [seq_len, n_vars]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]  # [pred_len, n_vars]

        return torch.FloatTensor(x), torch.FloatTensor(y)


class TimeMeTrainer:
    """Time-me模型训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # 创建模型
        self.model = CoupledMambaFusion(
            config=config,
            vision_dim=512,
            text_dim=512,
            d_model=config.d_model
        ).to(self.device)

        # 设置优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 创建数据加载器
        self.train_loader = self._create_data_loader('train')
        self.val_loader = self._create_data_loader('val')
        self.calib_loader = self._create_data_loader('calib')
        self.test_loader = self._create_data_loader('test')

        # 训练日志
        self.train_log = []
        self.best_val_loss = float('inf')

    def _create_data_loader(self, split):
        """创建数据加载器"""
        dataset = TrafficDataset(
            data_path=self.config.data_path,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            split=split,
            restrict_vars=getattr(self.config, 'restrict_vars', -1),
            train_ratio=getattr(self.config, 'train_ratio', 0.7),
            val_ratio=getattr(self.config, 'val_ratio', 0.1),
            calib_ratio=getattr(self.config, 'calib_ratio', 0.1),
            test_ratio=getattr(self.config, 'test_ratio', 0.1),
        )

        batch_size = self.config.batch_size if split == 'train' else self.config.batch_size * 2

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=True
        )

    def _create_multimodal_features(self, x_enc):
        """创建多模态特征（模拟VLM输出）"""
        batch_size, seq_len, n_vars = x_enc.shape

        # 时序特征 - 使用原始数据
        temporal_features = x_enc  # [B, seq_len, n_vars]

        # 视觉特征 - 模拟VLM图像编码（按模块期望的 [B,1,dim] 形状）
        vision_features = torch.randn(batch_size, 1, 512, device=x_enc.device) * 0.1

        # 文本特征 - 模拟VLM文本编码（按模块期望的 [B,dim] 形状）
        text_features = torch.randn(batch_size, 512, device=x_enc.device) * 0.1

        return temporal_features, vision_features, text_features

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        start_time = time.time()

        for batch_idx, (x_enc, y_true) in enumerate(self.train_loader):
            x_enc, y_true = x_enc.to(self.device), y_true.to(self.device)

            self.optimizer.zero_grad()

            # 创建多模态特征
            temporal_features, vision_features, text_features = self._create_multimodal_features(x_enc)

            # 前向传播
            predictions = self.model(
                temporal_features=temporal_features,
                vision_embeddings=vision_features,
                text_embeddings=text_features
            )

            # 计算损失
            loss = self.criterion(predictions, y_true.transpose(1, 2))  # [B, n_vars, pred_len] vs [B, pred_len, n_vars]

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time

        return avg_loss, epoch_time

    def validate(self):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for x_enc, y_true in self.val_loader:
                x_enc, y_true = x_enc.to(self.device), y_true.to(self.device)

                # 创建多模态特征
                temporal_features, vision_features, text_features = self._create_multimodal_features(x_enc)

                # 前向传播
                predictions = self.model(
                    temporal_features=temporal_features,
                    vision_embeddings=vision_features,
                    text_embeddings=text_features
                )

                # 计算损失
                loss = self.criterion(predictions, y_true.transpose(1, 2))

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def test(self):
        """测试模型性能"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_mape = 0
        num_batches = 0
        # conformal accumulators
        conf_enabled = getattr(self.config, 'conformal_enable', False)
        conf_loaded = False
        if conf_enabled:
            calib_path = os.path.join('checkpoints', 'conformal_calib.npz')
            if os.path.exists(calib_path):
                try:
                    z = np.load(calib_path)
                    saved_mode = str(z['mode']) if 'mode' in z.files else 'static'
                    s_val = z['scale']
                    lam_hat = float(z['lam_hat']) if saved_mode != 'online' else None
                    online_calib = (
                        AdaptiveOnlineConformalCalibrator.from_npz(z)
                        if saved_mode == 'online'
                        else None
                    )
                    conf_loaded = True
                    exceed_sum = 0
                    width_sum = 0.0
                    count_sum = 0
                    lower_list, upper_list = [], []
                    pred_list, true_list = [], []
                    lambda_hist, alpha_hist = [], []
                except Exception as e:
                    print(f"[Conformal] failed to load calibration: {e}")

        with torch.no_grad():
            for x_enc, y_true in self.test_loader:
                x_enc, y_true = x_enc.to(self.device), y_true.to(self.device)

                # 创建多模态特征
                temporal_features, vision_features, text_features = self._create_multimodal_features(x_enc)

                # 前向传播
                predictions = self.model(
                    temporal_features=temporal_features,
                    vision_embeddings=vision_features,
                    text_embeddings=text_features
                )

                # 计算损失
                loss = self.criterion(predictions, y_true.transpose(1, 2))

                # 计算MAE
                mae = torch.mean(torch.abs(predictions - y_true.transpose(1, 2)))

                # 计算MAPE (避免除零)
                mape = torch.mean(torch.abs((predictions - y_true.transpose(1, 2)) / (y_true.transpose(1, 2) + 1e-8)))

                total_loss += loss.item()
                total_mae += mae.item()
                total_mape += mape.item()
                num_batches += 1

                # conformal metrics and collect intervals
                if conf_loaded:
                    pred_np = predictions.detach().cpu().numpy()
                    true_np = y_true.transpose(1, 2).detach().cpu().numpy()
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

        results = {
            'mse': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'mape': total_mape / num_batches
        }

        # save conformal metrics and intervals
        if conf_loaded and count_sum > 0:
            miscov = exceed_sum / count_sum
            avg_width = width_sum / count_sum
            os.makedirs('results', exist_ok=True)
            with open(os.path.join('results', 'conformal_metrics.txt'), 'a') as f:
                if saved_mode == 'online' and online_calib is not None:
                    f.write(
                        f"mode=online, alpha={getattr(self.config,'conformal_alpha',0.10):.4f}, "
                        f"final_mean_alpha={float(np.mean(online_calib.alpha_t_)):.6f}, "
                        f"final_mean_lambda={float(np.mean(online_calib.lambda_t_)):.6f}, "
                        f"miscoverage={miscov:.6f}, avg_width={avg_width:.6f}\n"
                    )
                else:
                    f.write(
                        f"mode=static, alpha={getattr(self.config,'conformal_alpha',0.10):.4f}, "
                        f"lam_hat={lam_hat:.6f}, miscoverage={miscov:.6f}, avg_width={avg_width:.6f}\n"
                    )
            os.makedirs('predictions', exist_ok=True)
            lower_all = np.concatenate(lower_list, axis=0)
            upper_all = np.concatenate(upper_list, axis=0)
            np.save(os.path.join('predictions', 'time_me_interval_lower.npy'), lower_all)
            np.save(os.path.join('predictions', 'time_me_interval_upper.npy'), upper_all)
            # also save predictions and truths for visualization
            preds_all = np.concatenate(pred_list, axis=0)
            trues_all = np.concatenate(true_list, axis=0)
            np.save(os.path.join('predictions', 'time_me_pred.npy'), preds_all)
            np.save(os.path.join('predictions', 'time_me_true.npy'), trues_all)
            if saved_mode == 'online' and lambda_hist:
                np.save(os.path.join('predictions', 'time_me_lambda_history.npy'), np.stack(lambda_hist, axis=0))
            if saved_mode == 'online' and alpha_hist:
                np.save(os.path.join('predictions', 'time_me_alpha_history.npy'), np.stack(alpha_hist, axis=0))

        return results

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }

        # 保存最新检查点
        checkpoint_path = f"checkpoints/time_me_checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = "checkpoints/time_me_best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with val_loss: {val_loss:.6f}")

    def train(self):
        """完整训练流程"""
        print(f"Starting Time-me training on {self.config.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # 创建检查点目录
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(self.config.epochs):
            # 训练
            train_loss, train_time = self.train_epoch(epoch)

            # 验证
            val_loss = self.validate()

            # 更新学习率
            self.scheduler.step()

            # 记录日志
            log_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr'],
                'train_time': train_time
            }
            self.train_log.append(log_entry)

            # 打印进度
            print(f"Epoch {epoch}/{self.config.epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                  f"Time: {train_time:.1f}s")

            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)

        # 最终测试
        print("\nTraining completed! Running final test...")
        # Conformal calibration on validation set
        if getattr(self.config, 'conformal_enable', False):
            try:
                best_path = os.path.join('checkpoints', 'time_me_best_model.pth')
                if os.path.exists(best_path):
                    best = torch.load(best_path, map_location=self.device)
                    self.model.load_state_dict(best['model_state_dict'])
                self.model.eval()
                preds_list, trues_list = [] , []
                with torch.no_grad():
                    for x_enc, y_true in self.calib_loader:
                        x_enc, y_true = x_enc.to(self.device), y_true.to(self.device)
                        temporal_features, vision_features, text_features = self._create_multimodal_features(x_enc)
                        pred = self.model(
                            temporal_features=temporal_features,
                            vision_embeddings=vision_features,
                            text_embeddings=text_features
                        )
                        preds_list.append(pred.detach().cpu().numpy())
                        trues_list.append(y_true.transpose(1, 2).detach().cpu().numpy())
                if len(preds_list) > 0:
                    preds = np.concatenate(preds_list, axis=0)
                    trues = np.concatenate(trues_list, axis=0)
                    residuals = np.abs(trues - preds)  # [N, n_vars, pred_len]
                    scale_method = getattr(self.config, 'conformal_scale', 'mad')
                    s_val = estimate_scale_from_residuals(residuals, scale_method)
                    max_th = getattr(self.config, 'conformal_max_threshold', -1.0)
                    max_th = None if (max_th is None or max_th <= 0) else float(max_th)
                    mode = getattr(self.config, 'conformal_mode', 'online')
                    save_payload = {
                        'scale': s_val,
                        'method': getattr(self.config, 'conformal_method', 'crc'),
                        'alpha': getattr(self.config, 'conformal_alpha', 0.10),
                        'scale_method': scale_method,
                    }
                    if mode == 'online':
                        calib = AdaptiveOnlineConformalCalibrator(
                            alpha=getattr(self.config, 'conformal_alpha', 0.10),
                            update_lr=getattr(self.config, 'conformal_update_lr', 0.005),
                            buffer_size=getattr(self.config, 'conformal_buffer_size', 4096),
                            alpha_min=getattr(self.config, 'conformal_alpha_min', 0.005),
                            alpha_max=getattr(self.config, 'conformal_alpha_max', 0.30),
                            pred_len=getattr(self.config, 'pred_len', None),
                            threshold_method=getattr(self.config, 'conformal_method', 'hpd'),
                            recompute_method=getattr(self.config, 'conformal_recompute_method', 'bq'),
                            hpd_level=getattr(self.config, 'conformal_hpd_level', 0.95),
                            num_dir=getattr(self.config, 'conformal_num_dir', 1000),
                            delta=getattr(self.config, 'conformal_delta', 0.05),
                            max_threshold=None
                            if getattr(self.config, 'conformal_max_threshold', -1.0) <= 0
                            else float(getattr(self.config, 'conformal_max_threshold', -1.0)),
                            update_block_size=getattr(self.config, 'conformal_update_block_size', 8),
                            rng_seed=getattr(self.config, 'seed', 0),
                        )
                        calib.fit(residuals, s_val)
                        save_payload.update(calib.state_dict(s_val))
                        np.savez(os.path.join('checkpoints', 'conformal_calib.npz'), **save_payload)
                        print(
                            "[Conformal] initialized online calibration on calib split, "
                            f"mean_lambda={float(np.mean(calib.lambda_t_)):.4f}"
                        )
                    else:
                        calib = ConformalCalibrator(
                            method=getattr(self.config, 'conformal_method', 'crc'),
                            alpha=getattr(self.config, 'conformal_alpha', 0.10),
                            hpd_level=getattr(self.config, 'conformal_hpd_level', 0.95),
                            num_dir=getattr(self.config, 'conformal_num_dir', 1000),
                            delta=getattr(self.config, 'conformal_delta', 0.05),
                        )
                        calib.fit(residuals, s_val, lam_hi=max_th)
                        save_payload.update({'mode': np.array('static'), 'lam_hat': calib.lam_hat_})
                        np.savez(os.path.join('checkpoints', 'conformal_calib.npz'), **save_payload)
                        print(f"[Conformal] calibrated on calib split, lam_hat={calib.lam_hat_:.4f}")
            except Exception as e:
                print(f"[Conformal] calibration skipped due to error: {e}")
        test_metrics = self.test()

        print(f"\nFinal Test Results:")
        print(f"MSE: {test_metrics['mse']:.6f}")
        print(f"MAE: {test_metrics['mae']:.6f}")
        print(f"MAPE: {test_metrics['mape']:.6f}")

        # 保存训练日志
        with open("checkpoints/training_log.json", 'w') as f:
            json.dump(self.train_log, f, indent=2)

        return test_metrics


def main():
    """主函数：支持命令行参数与保序校准开关"""
    args = get_args()
    config = create_config_from_args(args)

    # 推导数据路径（root_path/data.csv）
    data_csv = os.path.join(args.root_path, args.data + '.csv')
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"dataset CSV not found: {data_csv}")
    config.data_path = data_csv

    # 必要默认值（如未由 CLI 指定）
    if not hasattr(config, 'weight_decay'):
        config.weight_decay = 1e-5

    # 创建训练器
    trainer = TimeMeTrainer(config)

    # 开始训练
    start_time = time.time()
    test_metrics = trainer.train()
    total_time = time.time() - start_time

    print(f"\nTraining completed in {total_time/60:.1f} minutes")

    return test_metrics


if __name__ == "__main__":
    test_metrics = main()
