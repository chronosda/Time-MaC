import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional, Tuple
from functools import partial

# Try to import Mamba; provide a safe fallback if unavailable
try:
    from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
    from mamba_ssm.modules.block import Block as MambaBlock  # type: ignore
    try:
        from mamba_ssm.modules.mlp import GatedMLP as MambaMLP  # type: ignore
    except Exception:
        MambaMLP = None
    mamba_available = True
except Exception:
    Mamba, MambaBlock, MambaMLP = None, None, None
    mamba_available = False

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn  # type: ignore
except Exception:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class FusionNet(nn.Module):
    """FusionNet implementation matching original coupled-mamba interface"""
    def __init__(self, dim: int, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.hidden_dim = output_dim  # Project to Mamba's expected dimension

        # Input projection
        self.input_proj = nn.Linear(dim, self.hidden_dim)

        # Self-attention mechanism (batch_first for [B, L, D])
        self.attention = nn.MultiheadAttention(self.hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        )
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x, memory=None):
        # Input projection
        x = self.input_proj(x)

        # Self-attention (handle sequence dimension properly)
        if x.dim() == 2:
            # Add sequence dimension if missing: [B, D] -> [B, 1, D]
            x = x.unsqueeze(1)

        # Apply attention
        x2 = self.norm1(x)
        if memory is not None:
            if memory.dim() == 2:
                memory = memory.unsqueeze(1)
            memory_proj = self.input_proj(memory)
            x2, _ = self.attention(x2, memory_proj, memory_proj)
        else:
            x2, _ = self.attention(x2, x2, x2)
        x = x + self.dropout(x2)

        # Feed-forward
        x2 = self.norm2(x)
        x2 = self.ff(x2)
        x = x + self.dropout(x2)

        # Output projection
        x = self.output_proj(x)

        return x


class ResNetBranch(nn.Module):
    """Cross-modal enhancement branch"""
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class RegressionHead(nn.Module):
    """Enhanced regression head for time series forecasting"""
    def __init__(self, d_model: int, pred_len: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(d_model // 2, pred_len)

    def forward(self, features):
        x = self.dense(features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class _FakeBlock(nn.Module):
    """Fallback block using TransformerEncoderLayer when Mamba is unavailable."""
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)

    def forward(self, x):
        return (self.layer(x),)


def create_block_enhanced(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """Enhanced block creation for coupled-mamba"""
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if mamba_available and Mamba is not None and MambaBlock is not None:
        mixer_cls = Mamba
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        # Try preferred signature including mlp_cls; fallback to legacy if needed
        try:
            mlp_cls = MambaMLP if MambaMLP is not None else None
            if mlp_cls is None:
                raise TypeError('No MLP available')
            block = MambaBlock(
                d_model,
                mixer_cls,
                mlp_cls,
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
            )
        except TypeError:
            block = MambaBlock(
                d_model,
                mixer_cls,
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
            )
        block.layer_idx = layer_idx
        return block
    # Fallback
    return _FakeBlock(d_model)


class EnhancedMultiMamba(nn.Module):
    """Fixed MultiMamba implementation for Time-VLM integration"""
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        pred_len: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.n_layer = n_layer
        self.pred_len = pred_len

        # Fixed modality projection layers - project all to d_model for Mamba
        self.audio_projection = FusionNet(audio_dim, d_model).to(device)
        self.vision_projection = FusionNet(vision_dim, d_model).to(device)
        self.text_projection = FusionNet(text_dim, d_model).to(device)

        # Cross-modal enhancement branches
        self.resnet_audio = ResNetBranch(d_model).to(device)
        self.resnet_vision = ResNetBranch(d_model).to(device)
        self.resnet_text = ResNetBranch(d_model).to(device)

        # Core network - three parallel Mamba branches for multimodal processing
        self.mamba_branches = nn.ModuleList([
            nn.ModuleList([
                create_block_enhanced(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs
                ) for i in range(n_layer)
            ]) for _ in range(3)  # Three modality branches
        ])

        # Normalization layer
        norm_class = nn.LayerNorm if not rms_norm else RMSNorm
        self.norm_f = norm_class(d_model, eps=norm_epsilon, **factory_kwargs)

        # Learnable fusion weights
        self.weight_audio = nn.Parameter(torch.ones(1))
        self.weight_visual = nn.Parameter(torch.ones(1))
        self.weight_text = nn.Parameter(torch.ones(1))

        # Output layer for time series forecasting
        self.regression_head = RegressionHead(d_model, pred_len).to(device)

        # Initialize weights
        for branch in self.mamba_branches:
            branch.apply(self._init_weights)

    def _init_weights(self, module, n_layer=None):
        """Enhanced weight initialization"""
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_audio, input_vision, input_text):
        """
        Enhanced forward pass for multimodal fusion

        Args:
            input_audio: Temporal features [B, n_vars, d_model]
            input_vision: Vision embeddings [B, 1, vision_dim]
            input_text: Text embeddings [B, text_dim]

        Returns:
            Fused predictions [B, n_vars, pred_len]
        """
        B, seq_len, d_model = input_audio.shape
        n_vars = seq_len  # number of variables equals sequence length dimension here

        # Feature projection - follow original coupled-mamba approach
        audio_emb = self.audio_projection(input_audio)
        vision_emb = self.vision_projection(input_vision.squeeze(1))
        text_emb = self.text_projection(input_text)

        embeddings = [audio_emb, vision_emb, text_emb]

        # Layer-wise processing - follow original coupled-mamba exactly
        for layer_idx in range(self.n_layer):
            branch_outputs = []

            # Independent branch processing
            for branch_idx, branch_layers in enumerate(self.mamba_branches):
                hidden_state = branch_layers[layer_idx](embeddings[branch_idx])
                if isinstance(hidden_state, (tuple, list)):
                    hidden_state = hidden_state[0]
                normalized_state = self.norm_f(hidden_state.to(self.norm_f.weight.dtype))
                branch_outputs.append(normalized_state + embeddings[branch_idx])

            # Final layer processing
            if layer_idx == self.n_layer - 1:
                pooled_outputs = []
                for output in branch_outputs:
                    pooled, _ = torch.max(output, dim=1)
                    pooled_outputs.append(pooled)

                # Weighted fusion
                weights = torch.stack([
                    self.weight_audio,
                    self.weight_visual,
                    self.weight_text
                ])
                norm_weights = torch.softmax(weights, dim=0)

                fused_embedding = sum(
                    w * output for w, output in zip(norm_weights, pooled_outputs)
                )
                break

            # Cross-modal enhancement - follow original coupled-mamba exactly
            # Align sequence dims for cross-modal enhancement
            n_vars = branch_outputs[0].shape[1]
            vision_b = branch_outputs[1]
            text_b = branch_outputs[2]
            # Broadcast vision/text to audio seq length for audio enhancement
            vision_broadcast = vision_b.expand(-1, n_vars, -1)
            text_broadcast = text_b.expand(-1, n_vars, -1)
            audio_enhanced = self.resnet_audio(
                torch.cat([vision_broadcast, text_broadcast], dim=-1)
            )
            # Pool audio to single step for vision/text enhancement
            audio_pooled = branch_outputs[0].mean(dim=1, keepdim=True)
            vision_enhanced = self.resnet_vision(
                torch.cat([audio_pooled, text_b], dim=-1)
            )
            text_enhanced = self.resnet_text(
                torch.cat([audio_pooled, vision_b], dim=-1)
            )

            # Update embeddings
            embeddings = [
                branch_outputs[0] + audio_enhanced,
                branch_outputs[1] + vision_enhanced,
                branch_outputs[2] + text_enhanced
            ]

        # Final prediction through regression head
        predictions = self.regression_head(fused_embedding)

        # Reshape to match Time-VLM output format [B, n_vars, pred_len]
        if predictions.dim() == 3 and predictions.shape[1] == seq_len:
            # If predictions have sequence dimension, pool across sequence
            predictions = predictions.mean(dim=1)  # [B, seq_len, pred_len] -> [B, pred_len]

        if predictions.dim() == 2:
            # Expand to [B, n_vars, pred_len] using detected n_vars
            predictions = predictions.unsqueeze(1).expand(-1, n_vars, -1)

        return predictions


class CoupledMambaFusion(nn.Module):
    """Main Coupled Mamba fusion module for Time-VLM integration"""
    def __init__(self, config, vision_dim, text_dim, d_model):
        super().__init__()
        self.config = config
        self.d_model = d_model
        # Pre-projection to map raw variable dimension to model dim per time step
        in_vars = getattr(config, 'restrict_vars', -1)
        if in_vars is None or in_vars <= 0:
            in_vars = getattr(config, 'enc_in', d_model)
        self.pre_proj = nn.Linear(in_vars, d_model)

        # Enhanced MultiMamba for multimodal fusion
        self.multimamba = EnhancedMultiMamba(
            d_model=d_model,
            n_layer=getattr(config, 'mamba_layers', 2),
            audio_dim=d_model,
            vision_dim=vision_dim,
            text_dim=text_dim,
            pred_len=config.pred_len,
            ssm_cfg={'d_state': 64, 'd_conv': 4, 'expand': 2},
            device=config.device if hasattr(config, 'device') else None
        )

    def forward(self, temporal_features, vision_embeddings, text_embeddings):
        """
        Forward pass for coupled-mamba fusion

        Args:
            temporal_features: [B, n_vars, d_model]
            vision_embeddings: [B, 1, vision_dim]
            text_embeddings: [B, text_dim]

        Returns:
            Fused predictions [B, n_vars, pred_len]
        """
        # Ensure temporal features have last dim = d_model via pre-projection
        if temporal_features.shape[-1] != self.d_model:
            temporal_features = self.pre_proj(temporal_features)
        return self.multimamba(
            input_audio=temporal_features,
            input_vision=vision_embeddings,
            input_text=text_embeddings
        )
