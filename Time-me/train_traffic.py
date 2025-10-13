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
from utils.conformal_plugin import ConformalCalibrator


class TrafficDataset(Dataset):
    """Traffic数据集加载器"""
    def __init__(self, data_path, seq_len=96, pred_len=12, split='train', restrict_vars: int = -1):
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

        # 标准化数据
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8
        data = (data - self.mean) / self.std

        # 划分训练集和测试集
        total_len = len(data)
        train_ratio = 0.7
        val_ratio = 0.15

        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))

        if split == 'train':
            self.data = data[:train_end]
        elif split == 'val':
            self.data = data[train_end:val_end]
        else:  # test
            self.data = data[val_end:]

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
            restrict_vars=getattr(self.config, 'restrict_vars', -1)
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
                    lam_hat = float(z['lam_hat'])
                    s_val = z['scale']
                    conf_loaded = True
                    exceed_sum = 0
                    width_sum = 0.0
                    count_sum = 0
                    lower_list, upper_list = [], []
                    pred_list, true_list = [], []
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
                    # broadcast scale to batch
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
                f.write(f"alpha={getattr(self.config,'conformal_alpha',0.10):.4f}, lam_hat={lam_hat:.6f}, "
                        f"miscoverage={miscov:.6f}, avg_width={avg_width:.6f}\n")
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
                    for x_enc, y_true in self.val_loader:
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
                    # scale estimation
                    scale_method = getattr(self.config, 'conformal_scale', 'mad')
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
                        s_val = np.full_like(residuals[0:1], s_scalar)  # [1, n_vars, pred_len]

                    max_th = getattr(self.config, 'conformal_max_threshold', -1.0)
                    max_th = None if (max_th is None or max_th <= 0) else float(max_th)
                    calib = ConformalCalibrator(
                        method=getattr(self.config, 'conformal_method', 'crc'),
                        alpha=getattr(self.config, 'conformal_alpha', 0.10),
                        hpd_level=getattr(self.config, 'conformal_hpd_level', 0.95),
                        num_dir=getattr(self.config, 'conformal_num_dir', 1000),
                        delta=getattr(self.config, 'conformal_delta', 0.05),
                    )
                    calib.fit(residuals, s_val, lam_hi=max_th)
                    np.savez(os.path.join('checkpoints', 'conformal_calib.npz'),
                             lam_hat=calib.lam_hat_, scale=s_val,
                             method=getattr(self.config, 'conformal_method', 'crc'),
                             alpha=getattr(self.config, 'conformal_alpha', 0.10),
                             scale_method=scale_method)
                    print(f"[Conformal] calibrated lam_hat={calib.lam_hat_:.4f}")
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
