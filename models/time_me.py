import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
from PIL import Image

# Import custom modules
from src.coupled_mamba_fusion import CoupledMambaFusion
from src.TimeVLM.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.TimeSeries_To_Image import time_series_to_simple_image
from models.timebase_adapter import build_timebase_for_dataset


class PatchMemoryBank(nn.Module):
    """Enhanced patch memory bank for Time-me"""
    def __init__(self, max_size, patch_size, feature_dim, device=None):
        super().__init__()
        self.max_size = max_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device('cpu')
        self.register_buffer('patches', torch.zeros((max_size, feature_dim), device=self.device))
        self.ptr = 0

    def update(self, new_patches):
        n = new_patches.size(0)
        new_patches_flat = new_patches.mean(dim=1)

        if self.patches.device != new_patches_flat.device:
            self.patches = self.patches.to(new_patches_flat.device)

        if self.ptr + n > self.max_size:
            self.patches[self.ptr:] = new_patches_flat[:self.max_size - self.ptr]
            self.ptr = 0
        else:
            self.patches[self.ptr:self.ptr + n] = new_patches_flat
            self.ptr += n

    def retrieve(self, query_patches, top_k=5):
        query_flat = query_patches.mean(dim=1)
        memory_flat = self.patches

        if query_flat.device != memory_flat.device:
            memory_flat = memory_flat.to(query_flat.device)

        similarity = torch.matmul(query_flat, memory_flat.T)
        _, indices = similarity.topk(top_k, dim=-1)

        if indices.device != self.patches.device:
            indices = indices.to(self.patches.device)

        retrieved_patches = self.patches[indices]
        return retrieved_patches, indices


class TimeMEModel(nn.Module):
    """
    Time-MaC/Time-me model with Coupled-Mamba fusion.
    Improved architecture addressing interface mismatches and architectural issues
    """
    def __init__(self, config, **kwargs):
        super(TimeMEModel, self).__init__()
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu}' if getattr(config, 'use_gpu', False) else 'cpu')

        # Use enhanced architecture flag
        self.use_enhanced_fusion = getattr(config, 'use_enhanced_fusion', True)
        # Optional: use TimeBase as temporal backbone / complexity reducer
        self.use_timebase_backbone = getattr(config, 'use_timebase_backbone', False)

        # Initialize VLM for text processing
        self.vlm_manager = VLMManager(config)
        # VLMManager is a lightweight controller rather than an nn.Module. Register
        # its model explicitly so trainable adapters are optimized and checkpointed.
        if self.vlm_manager.model is None:
            raise RuntimeError("The configured VLM failed to initialize.")
        self.vlm_model = self.vlm_manager.model

        # Vision processing path inherited from the VLM-compatible scaffold.
        vision_dim = self.vlm_manager.hidden_size

        # Initialize enhanced patch memory bank
        self.patch_memory_bank = PatchMemoryBank(
            max_size=getattr(config, 'patch_memory_size', 100),
            patch_size=config.patch_len,
            feature_dim=config.d_model,
            device=self.device
        )

        # Determine vision dimension
        vision_dim = self.vlm_manager.hidden_size

        # Initialize core modules
        self._init_modules(config, vision_dim)

    def _init_modules(self, config, vision_dim):
        """Initialize all model modules with proper configuration"""
        # Patch embedding for temporal features (default backbone)
        self.patch_embedding = PatchEmbedding(
            config.d_model,
            config.patch_len,
            config.stride,
            config.padding,
            config.dropout
        )

        # Calculate number of patches
        self.num_patches = int((config.seq_len + 2 * config.padding - config.patch_len) / config.stride + 1)
        self.head_nf = config.d_model * self.num_patches
        self.flatten = nn.Flatten(start_dim=-2)

        # Enhanced prediction heads for default backbone
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.dropout)
        )

        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.d_model),
            nn.Dropout(config.dropout)
        )

        # Optional TimeBase temporal backbone (complexity reducer)
        if self.use_timebase_backbone:
            period_len = getattr(config, 'period_len', getattr(config, 'periodicity', 24))
            basis_num = getattr(config, 'basis_num', 6)
            lambda_orth = float(getattr(config, 'lambda_orth', 0.0))
            self.timebase_backbone = build_timebase_for_dataset(
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                enc_in=getattr(config, 'enc_in', 1),
                period_len=period_len,
                basis_num=basis_num,
                lambda_orth=lambda_orth,
            ).to(self.device)
            # Project TimeBase predictions [B, pred_len, C] into temporal features
            self.timebase_proj = nn.Linear(config.pred_len * getattr(config, 'enc_in', 1), config.d_model)

        # Enhanced Coupled Mamba fusion module
        if self.use_enhanced_fusion:
            self.coupled_mamba_fusion = CoupledMambaFusion(
                config=config,
                vision_dim=vision_dim,
                text_dim=self.vlm_manager.hidden_size,
                d_model=config.d_model
            )
        else:
            # Fallback to the VLM-compatible feature path.
            self.coupled_mamba_fusion = None

        # Enhanced memory modules
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )

        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=getattr(config, 'num_attention_heads', 8),
            dropout=config.dropout,
            batch_first=True
        )

        # Keep original image converter as fallback
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3,
            hidden_dim=48,
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size,
            periodicity=config.periodicity
        )

        # Learnable gating parameters for enhanced fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.3))
        # Learnable fusion gate (memory vs. multimodal), sigmoid(logit) -> [0,1]
        # init to ~0.7 for memory branch: logit(0.7) ≈ 0.8473
        self.fusion_gate_logit = nn.Parameter(torch.tensor(0.84729786))
        self.layer_norm = nn.LayerNorm(config.d_model)

    def _compute_local_memory(self, patches):
        """Enhanced local memory computation with better retrieval"""
        device = patches.device
        patches = patches.to(device)

        # Enhanced retrieval with adaptive top-k
        top_k = min(getattr(self.config, 'top_k', 5), self.patch_memory_bank.max_size)
        retrieved_patches, _ = self.patch_memory_bank.retrieve(patches, top_k=top_k)

        retrieved_patches = retrieved_patches.to(device)

        # Process retrieved patches with enhanced MLP
        local_memory = self.local_memory_mlp(retrieved_patches)

        # Adaptive averaging with attention weights
        attention_weights = torch.softmax(torch.matmul(patches, retrieved_patches.transpose(-2, -1)), dim=-1)
        local_memory = torch.sum(attention_weights.unsqueeze(-1) * retrieved_patches.unsqueeze(1), dim=2)

        # Residual connection with gating
        local_memory = self.alpha * local_memory + (1 - self.alpha) * patches

        return local_memory

    def _compute_global_memory(self, patches):
        """Enhanced global memory computation with multi-head attention"""
        # Enhanced self-attention with multi-head mechanism
        attn_output, attn_weights = self.memory_attention(
            query=patches,
            key=patches,
            value=patches
        )

        # Temporal pooling across patch dimension
        global_memory = attn_output.mean(dim=1, keepdim=True)

        # Update patch memory bank with current patches
        self.patch_memory_bank.update(patches.detach())

        return global_memory

    def forward_prediction(self, x_enc, vision_embeddings, text_embeddings):
        """Enhanced forward prediction with coupled-mamba fusion"""
        B, L, n_vars = x_enc.shape

        # 1. Process temporal features: either default patch-based backbone or TimeBase backbone
        if self.use_timebase_backbone:
            # TimeBase backbone: low-complexity temporal modeling
            tb_out = self.timebase_backbone(x_enc)
            if isinstance(tb_out, tuple):
                tb_pred, _ = tb_out  # ignore orthogonal loss here; handled inside backbone
            else:
                tb_pred = tb_out
            # tb_pred: [B, pred_len, C]
            Btb, Ltb, Ctb = tb_pred.shape
            # Project flattened TimeBase predictions into d_model, then broadcast per variable
            time_feat = tb_pred.reshape(Btb, Ltb * Ctb)
            temporal_vec = self.timebase_proj(time_feat)  # [B, d_model]
            temporal_features = temporal_vec.unsqueeze(1).repeat(1, n_vars, 1)  # [B, n_vars, d_model]
            # Use TimeBase predictions directly as memory features [B, n_vars, pred_len]
            memory_features = tb_pred.permute(0, 2, 1)  # [B, C, pred_len] -> treat C == n_vars
        else:
            device = x_enc.device
            patches, _ = self.patch_embedding(x_enc.transpose(1, 2).to(device))

            # Enhanced local and global memory computation
            local_memory = self._compute_local_memory(patches)
            global_memory = self._compute_global_memory(patches)

            # Enhanced memory combination with learnable weights
            global_memory_expanded = global_memory.expand(-1, local_memory.shape[1], -1)
            memory_features = self.beta * local_memory + (1 - self.beta) * global_memory_expanded

            # Enhanced temporal processing
            memory_features_flat = self.flatten(memory_features)

            # Dynamic adjustment for varying input sizes
            actual_head_nf = memory_features_flat.shape[-1]
            if actual_head_nf != self.head_nf:
                if not hasattr(self, 'temporal_head_adjusted'):
                    self.temporal_head = nn.Sequential(
                        nn.Linear(actual_head_nf, self.config.d_model),
                        nn.Dropout(self.config.dropout)
                    ).to(memory_features_flat.device)
                    self.memory_head = nn.Sequential(
                        nn.Linear(actual_head_nf, self.config.pred_len),
                        nn.Dropout(self.config.dropout)
                    ).to(memory_features_flat.device)
                    self.temporal_head_adjusted = True

            temporal_features = self.temporal_head(memory_features_flat)
            memory_features = self.memory_head(memory_features_flat)

            # Reshape for multimodal fusion
            temporal_features = einops.rearrange(temporal_features, '(b n) d -> b n d', b=B, n=n_vars)
            memory_features = einops.rearrange(memory_features, '(b n) d -> b n d', b=B, n=n_vars)

        # 5. Enhanced coupled-mamba fusion
        if self.coupled_mamba_fusion is not None:
            # Ensure all inputs are on the same device
            device = temporal_features.device
            multimodal_predictions = self.coupled_mamba_fusion(
                temporal_features=temporal_features.to(device),
                vision_embeddings=vision_embeddings.to(device),
                text_embeddings=text_embeddings.to(device)
            )

            # Enhanced combination with learnable gating (sigmoid gate)
            gate = torch.sigmoid(self.fusion_gate_logit)
            predictions = gate * memory_features + (1.0 - gate) * multimodal_predictions
        else:
            # Fallback to memory-only predictions
            predictions = memory_features

        return predictions.permute(0, 2, 1)

    def _normalize_input(self, x_enc):
        """Enhanced input normalization"""
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    def _denormalize_output(self, output, means, stdev):
        """Enhanced output denormalization"""
        return output * stdev + means

    def _generate_prompts(self, x_enc):
        """Build Time-VLM-style prompts from dataset context and sample statistics."""
        batch_size, seq_len, n_vars = x_enc.shape
        description = getattr(self.config, 'dataset_description', '').strip()
        if not description:
            raise ValueError(
                "dataset_description is empty; load a dataset prompt before creating the model"
            )

        prompts = []
        for i in range(batch_size):
            sample = x_enc[i].detach()
            min_value = sample.min().item()
            max_value = sample.max().item()
            median_value = sample.median().item()
            trend = sample.diff(dim=0).sum().item()
            trend_direction = "upward" if trend > 0 else "downward"
            prompt = (
                f"Dataset background: {description} "
                f"Task: Forecast the next {self.config.pred_len} steps from the past "
                f"{seq_len} steps of {n_vars} variables. "
                f"Input statistics: minimum {min_value:.3f}, maximum {max_value:.3f}, "
                f"median {median_value:.3f}, and overall trend {trend_direction}."
            )
            prompts.append(prompt)
        return prompts

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """Enhanced forward pass"""
        B, L, D = x_enc.shape

        # Enhanced input normalization
        x_enc, means, stdev = self._normalize_input(x_enc)

        # Enhanced prompt generation
        prompts = self._generate_prompts(x_enc)

        # Vision processing path inherited from the VLM-compatible scaffold.
        device = x_enc.device
        images = self.learnable_image_module(x_enc.to(device))
        # Ensure 3 channels for VLMs like CLIP
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        # Normalize to [0, 1] per image and convert to PIL list
        images = images.float()
        min_vals = images.amin(dim=(1, 2, 3), keepdim=True)
        max_vals = images.amax(dim=(1, 2, 3), keepdim=True)
        images = (images - min_vals) / (max_vals - min_vals + 1e-5)
        if self.config.vlm_type.lower() == 'mae':
            # Keep the tensor path differentiable for the learnable image converter.
            vlm_images = images
        else:
            vlm_images = []
            for i in range(images.shape[0]):
                arr = (images[i].detach().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype('uint8')
                vlm_images.append(Image.fromarray(arr))
        batch_prompts = [prompts[i] for i in range(B)]
        vision_embeddings, text_embeddings = self.vlm_manager.process_inputs(B, vlm_images, batch_prompts)
        vision_embeddings = vision_embeddings.unsqueeze(1)  # [B, 1, hidden_size]

        # Enhanced prediction
        predictions = self.forward_prediction(x_enc, vision_embeddings, text_embeddings)

        # Enhanced output denormalization
        predictions = self._denormalize_output(predictions, means, stdev)

        return predictions

    def train(self, mode=True):
        """Enhanced training mode with proper device handling"""
        super().train(mode)
        return self

    def eval(self):
        """Enhanced evaluation mode"""
        super().eval()
        return self

    def get_model_info(self):
        """Get model information for debugging"""
        info = {
            'model_type': 'TimeME (Time-MaC with Coupled-Mamba)',
            'use_enhanced_fusion': self.use_enhanced_fusion,
            'use_mae_vision': getattr(self.config, 'use_mae_vision', False),
            'd_model': self.config.d_model,
            'pred_len': self.config.pred_len,
            'seq_len': self.config.seq_len,
            'n_vars': getattr(self.config, 'enc_in', 1),
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return info
