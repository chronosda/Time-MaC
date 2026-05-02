#!/usr/bin/env python3
"""
Synthetic multimodal experiment for Time-me / Coupled-Mamba.

Purpose
-------
Construct a small synthetic dataset where:
  - temporal (audio) features alone are not sufficient to fully predict the target;
  - vision/text modalities carry additional predictive information;
and compare:
  (1) an audio-only baseline;
  (2) CoupledMambaFusion using all three modalities.

This script is designed to be lightweight and CPU-friendly by default.
You can increase dataset size / epochs to obtain clearer gaps if needed.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

from src.coupled_mamba_fusion import CoupledMambaFusion


@dataclass
class SyntheticConfig:
    d_model: int = 64
    pred_len: int = 16
    seq_len: int = 64
    enc_in: int = 4
    c_out: int = 4
    mamba_layers: int = 2
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    # mamba ssm config (used inside CoupledMambaFusion)
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2


class SyntheticMultimodalDataset(Dataset):
    """
    Synthetic dataset with three modalities:
      - temporal series x(t) in R^{enc_in}
      - vision embedding v in R^{vision_dim}
      - text embedding z in R^{text_dim}
    The target y depends on all three in a controlled way.
    """

    def __init__(
        self,
        num_samples: int = 4096,
        seq_len: int = 64,
        pred_len: int = 16,
        enc_in: int = 4,
        vision_dim: int = 64,
        text_dim: int = 64,
        seed: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        rng = np.random.default_rng(seed)

        # time grid
        t_all = np.linspace(0, 4 * math.pi, seq_len + pred_len, dtype=np.float32)

        xs = []
        vs = []
        zs = []
        ys = []

        for _ in range(num_samples):
            # temporal pattern: mixture of sines with random phase per channel
            base = []
            for c in range(enc_in):
                freq1 = 0.5 + 0.1 * c
                freq2 = 0.3 + 0.05 * c
                phase = rng.uniform(0, 2 * math.pi)
                series = (
                    np.sin(freq1 * t_all + phase)
                    + 0.3 * np.sin(freq2 * t_all)
                    + 0.1 * rng.standard_normal(t_all.shape)
                )
                base.append(series.astype(np.float32))
            base = np.stack(base, axis=-1)  # [T, enc_in]

            x_enc = base[:seq_len, :]  # encoder part

            # vision embedding carries a "future amplitude" hint
            future_segment = base[seq_len:, :].mean(axis=0)  # [enc_in]
            v = np.tanh(np.concatenate([future_segment, future_segment * 0.5]))  # [2*enc_in]
            v = v.astype(np.float32)
            if v.shape[0] < vision_dim:
                v = np.pad(v, (0, vision_dim - v.shape[0]))
            else:
                v = v[:vision_dim]

            # text embedding carries a "frequency" hint via random projection
            freq_hint = np.array(
                [np.sin(0.1 * t_all[-1]), np.cos(0.15 * t_all[-1])], dtype=np.float32
            )
            z = np.repeat(freq_hint, text_dim // 2)
            if z.shape[0] < text_dim:
                z = np.pad(z, (0, text_dim - z.shape[0]))
            else:
                z = z[:text_dim]

            # target: weighted combination of future temporal + v + z
            y_true = base[seq_len : seq_len + pred_len, :]  # [pred_len, enc_in]
            # add controlled dependency on v and z
            y_shift = 0.1 * future_segment + 0.05 * freq_hint.mean()
            y_true = y_true + y_shift[None, :enc_in]

            xs.append(x_enc)
            vs.append(v)
            zs.append(z)
            ys.append(y_true)

        self.xs = torch.from_numpy(np.stack(xs, axis=0))  # [N, seq_len, enc_in]
        self.vs = torch.from_numpy(np.stack(vs, axis=0))  # [N, vision_dim]
        self.zs = torch.from_numpy(np.stack(zs, axis=0))  # [N, text_dim]
        self.ys = torch.from_numpy(np.stack(ys, axis=0))  # [N, pred_len, enc_in]

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        return self.xs[idx], self.vs[idx], self.zs[idx], self.ys[idx]


class AudioOnlyBaseline(nn.Module):
    """Simple baseline: flatten temporal features and predict future."""

    def __init__(self, seq_len: int, enc_in: int, pred_len: int):
        super().__init__()
        in_dim = seq_len * enc_in
        out_dim = pred_len * enc_in
        self.net = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim),
            nn.GELU(),
            nn.Linear(4 * in_dim, 2 * in_dim),
            nn.GELU(),
            nn.Linear(2 * in_dim, out_dim),
        )
        self.pred_len = pred_len
        self.enc_in = enc_in

    def forward(self, x_enc):
        # x_enc: [B, L, C]
        B, L, C = x_enc.shape
        out = self.net(x_enc.reshape(B, L * C))
        return out.reshape(B, self.pred_len, self.enc_in)


def train_one(model, loader, device, epochs: int = 5, desc: str = ""):
    model.to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"{desc} Epoch {ep+1}/{epochs}")
        for x_enc, v, z, y_true in pbar:
            x_enc = x_enc.to(device)
            v = v.to(device)
            z = z.to(device)
            y_true = y_true.to(device)

            optim.zero_grad()
            if isinstance(model, CoupledMambaFusion):
                # temporal_features: [B, n_vars, d_model] — here use a linear projection
                B, L, C = x_enc.shape
                # simple projection to d_model
                temp = x_enc  # we will pre-project inside CoupledMambaFusion
                vision = v.unsqueeze(1)  # [B, 1, vision_dim]
                text = z  # [B, text_dim]
                preds = model(temp, vision, text)  # [B, n_vars, pred_len]
                preds = preds.transpose(1, 2)  # [B, pred_len, n_vars]
            else:
                preds = model(x_enc)

            loss = criterion(preds, y_true)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

        print(f"[{desc}] Epoch {ep+1} avg loss = {total_loss / max(1, n_batches):.6f}")


def main():
    cfg = SyntheticConfig()
    device = torch.device(cfg.device)

    # dataset & loader
    vision_dim = 64
    text_dim = 64
    train_ds = SyntheticMultimodalDataset(
        num_samples=2048,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        enc_in=cfg.enc_in,
        vision_dim=vision_dim,
        text_dim=text_dim,
        seed=42,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # 1) Audio-only baseline
    baseline = AudioOnlyBaseline(cfg.seq_len, cfg.enc_in, cfg.pred_len)
    train_one(baseline, train_loader, device, epochs=5, desc="AudioOnlyBaseline")

    # 2) Coupled-Mamba multimodal model
    # Note: CoupledMambaFusion internally projects temporal features to d_model.
    class WrapperConfig:
        # minimal config object to satisfy CoupledMambaFusion
        def __init__(self, cfg):
            self.d_model = cfg.d_model
            self.pred_len = cfg.pred_len
            self.mamba_layers = cfg.mamba_layers
            self.device = cfg.device
            self.enc_in = cfg.enc_in
            self.restrict_vars = -1

    cm_config = WrapperConfig(cfg)
    cm_model = CoupledMambaFusion(
        config=cm_config,
        vision_dim=vision_dim,
        text_dim=text_dim,
        d_model=cfg.d_model,
    )
    train_one(cm_model, train_loader, device, epochs=5, desc="CoupledMambaFusion")

    print("Synthetic multimodal experiment finished.")


if __name__ == "__main__":
    main()

