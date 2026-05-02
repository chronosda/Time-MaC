# Real-Multimodal Experiment Design for Time-ME / Coupled-Mamba

本说明文档给出一个面向“真实多模态场景”（时间序列 + 图像和/或文本）的实验设计和实现骨架，用于在论文中展示 Coupled‑Mamba 在有模态互补信息时的优势。

## 1. 目标

- 构造一个实际任务，其中每个时间序列样本带有配套的图像或文本（或二者兼有），例如：
  - 电网 / 交通：站点/路段的地理拓扑图、路网图、站点照片；
  - 气象：对应时刻的雷达图、卫星云图；
  - 事件驱动序列：与节假日、重大事件相关的描述文本。
- 对比结构：
  1. 仅用时间序列的基线模型（不使用 VLM）；
  2. 时间序列 + VLM 特征，简单拼接 / MLP 融合；
  3. 时间序列 + VLM 特征 + Coupled‑Mamba（本项目方法）。
- 实验目标：证明在“外部模态携带互补信息”的设定下，结构 (3) 相比 (1)/(2) 在预测精度和鲁棒性上有显著优势。

## 2. 数据格式与组织建议

假设你构造的数据目录如下：

```text
dataset/real_multimodal/
  train.csv
  val.csv
  test.csv
  images/
    <sample_id>.jpg
    ...
```

- 每行样本包含：
  - `sample_id`：唯一标识；
  - 时间序列数值列（多变量）；
  - 对应图像文件名（例如 `image_path` 列，值为 `images/<sample_id>.jpg`）；
  - 文本描述（例如 `text` 列，可为空字符串表示“无文本模态”）。

## 3. 数据集类骨架

你可以新建一个数据集类，例如放在 `utils/data_loader_multimodal.py` 中（下面是伪代码骨架）：

```python
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class RealMultimodalTimeSeriesDataset(Dataset):
    def __init__(self, csv_path, image_root, seq_len, pred_len, split='train', scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len

        df = pd.read_csv(csv_path)
        # 假设有 sample_id, image_path, text, 以及一系列数值列
        self.sample_ids = df['sample_id'].values
        self.image_paths = df['image_path'].values
        self.texts = df['text'].fillna("").values

        # 只保留数值列作为时间序列特征
        num_df = df.select_dtypes(include=[np.number])
        self.data = num_df.values  # [T_total, n_features] 或按样本重排
        # TODO: 根据你的数据格式整理成 [N, T, C]
        # 并根据 split 划分 train/val/test

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # x_enc: [seq_len, C], y: [pred_len, C]
        x_enc = ...
        y = ...

        # 图像
        img_path = os.path.join(self.image_root, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')

        text = self.texts[idx]

        return x_enc, image, text, y
```

> 注：由于每个实际数据集格式不同，上面只给出结构骨架；你需要根据自己的 CSV 设计，填充 `"..."` 部分，以构造好 `x_enc` / `y` 的切片。

## 4. 训练脚本骨架（真实多模态）

可以新建一个脚本，如 `scripts/train_time_me_real_multimodal.py`。核心思路：

1. 使用 `RealMultimodalTimeSeriesDataset` 提供 `(x_enc, image, text, y)`。
2. 调用 `TimeMEModel`，但将内部的时间序列转图像模块替换为“直接使用外部图像”；或者使用 `VLMManager` 独立抽取视觉/文本特征后，与时间序列编码做 Coupled‑Mamba 融合。

伪代码骨架类似：

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import TimeMEConfig
from src.TimeVLM.vlm_manager import VLMManager
from src.coupled_mamba_fusion import CoupledMambaFusion
from utils.data_loader_multimodal import RealMultimodalTimeSeriesDataset


class RealMultimodalModel(nn.Module):
    def __init__(self, config, vision_dim, text_dim):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # 简单时间序列编码：线性层或小型 Transformer
        self.ts_encoder = nn.Linear(config.enc_in, config.d_model)

        self.coupled = CoupledMambaFusion(
            config=config,
            vision_dim=vision_dim,
            text_dim=text_dim,
            d_model=config.d_model,
        )

    def forward(self, x_enc, vision_emb, text_emb):
        # x_enc: [B, L, C] -> [B, L, d_model]
        temp = self.ts_encoder(x_enc)
        # vision_emb: [B, 1, vision_dim], text_emb: [B, text_dim]
        preds = self.coupled(temp, vision_emb, text_emb)  # [B, n_vars, pred_len]
        return preds.transpose(1, 2)  # [B, pred_len, n_vars]


def main():
    cfg = TimeMEConfig()
    cfg.vlm_type = 'clip'  # 或 blip2/vilt/custom
    cfg.offline = False    # 需要实际加载 VLM
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # Dataset & loader
    train_ds = RealMultimodalTimeSeriesDataset(
        csv_path='dataset/real_multimodal/train.csv',
        image_root='dataset/real_multimodal/images',
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        split='train',
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    # VLM 管理器：负责将 (image, text) -> (vision_emb, text_emb)
    vlm = VLMManager(cfg)
    vision_dim = vlm.hidden_size
    text_dim = vlm.hidden_size

    model = RealMultimodalModel(cfg, vision_dim=vision_dim, text_dim=text_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(cfg.epochs):
        model.train()
        for x_enc, image, text, y_true in train_loader:
            x_enc = x_enc.to(device).float()          # [B, L, C]
            y_true = y_true.to(device).float()        # [B, pred_len, C]

            # 通过 VLMManager 抽取多模态特征
            images = [img for img in image]           # 列表 of PIL
            prompts = list(text)
            vision_emb, text_emb = vlm.process_inputs(len(images), images, prompts)
            vision_emb = vision_emb.unsqueeze(1)      # [B, 1, hidden_size]

            preds = model(x_enc, vision_emb, text_emb)
            loss = criterion(preds, y_true)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch {epoch+1}, train loss = {loss.item():.6f}")


if __name__ == "__main__":
    main()
```

> 实际实现时，你可以拷贝上述骨架到 `scripts/train_time_me_real_multimodal.py`，根据自己的数据格式微调数据集类与训练循环。

## 5. 实验报告建议

在论文中，可以围绕以下几点组织“真实多模态实验”小节：

1. **任务描述与数据来源**：说明时间序列任务、外部图像/文本模态的来源和含义。  
2. **实验设置**：包括预测步长、特征维度、VLM 类型（CLIP/BLIP2/ViLT/MAE）、训练超参等。  
3. **对比结构**：
   - TS‑only（无多模态）；
   - TS + VLM（简单拼接或加权和）；
   - TS + VLM + Coupled‑Mamba（本文方法）。  
4. **结果表与分析**：给出各结构在 MSE/MAE/R2/DA 上的对比，并结合合成多模态实验，强调：
   - 在模态互补明显时，Coupled‑Mamba 相比简单融合有稳定优势；
   - 在模态冗余或无信息时，Coupled‑Mamba 不会严重拖累性能（参考 Electricity 纯时序结果），体现一定的“自适应关闭”能力。

通过这一套“合成 + 真实”多模态实验，你可以把当前在时间序列基准上观察到的“Coupled‑Mamba 作用有限”解释为结构适用范围的问题，而不是结构本身无效，并在真实多模态场景下展示它的实际价值。

