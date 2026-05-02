import torch
import torch.nn as nn
import sys
from typing import Tuple
import os
import torch.nn.functional as F

# Import MAE models
sys.path.append("../")
from layers.models_mae import *


class DualPathReconstructionVLM(nn.Module):
    """
    Dual-path VLM: combines a standard VLM path with an MAE reconstruction path.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self._acquire_device()

        # Path config
        self.use_vlm_path = getattr(config, 'use_vlm_path', True)
        self.use_reconstruction_path = getattr(config, 'use_reconstruction_path', True)
        self.path_fusion_strategy = getattr(config, 'path_fusion_strategy', 'adaptive')  # adaptive|fixed|learned

        # Reconstruction config
        self.mae_arch = getattr(config, 'mae_arch', 'mae_base')
        self.reconstruction_strength = getattr(config, 'reconstruction_strength', 0.7)

        # Init paths
        self._init_dual_paths()
        # Init fusion
        self._init_path_fusion()
        # Dims
        self._set_dimensions()

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
        return device

    def _init_dual_paths(self):
        if self.use_vlm_path:
            self.vlm_path = VLMFeaturePath(self.config)
        if self.use_reconstruction_path:
            self.reconstruction_path = MAEReconstructionPath(self.config)

    def _init_path_fusion(self):
        if not (self.use_vlm_path and self.use_reconstruction_path):
            return
        hidden_size = 768
        if self.path_fusion_strategy == 'adaptive':
            self.path_selector = AdaptivePathSelector(hidden_size)
            self.path_fusion = AdaptivePathFusion(hidden_size)
        elif self.path_fusion_strategy == 'learned':
            self.path_fusion = LearnedPathFusion(hidden_size)
        else:
            self.path_fusion = FixedPathFusion(
                vlm_weight=1.0 - self.reconstruction_strength,
                reconstruction_weight=self.reconstruction_strength
            )

    def _set_dimensions(self):
        self.hidden_size = 768
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77
        self.fused_feature_len = self.hidden_size

    def forward(self, images, texts=None):
        batch_size = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
        vlm_features = torch.zeros(batch_size, self.hidden_size).to(self.device)
        if self.use_vlm_path:
            vlm_features, _ = self.vlm_path(images, texts)
        reconstruction_features = torch.zeros(batch_size, self.hidden_size).to(self.device)
        if self.use_reconstruction_path:
            reconstruction_features, reconstruction_info = self.reconstruction_path(images)
        if self.use_vlm_path and self.use_reconstruction_path:
            if self.path_fusion_strategy == 'adaptive':
                fusion_weight = self.path_selector(reconstruction_info)
                fused_features = self.path_fusion(vlm_features, reconstruction_features, fusion_weight)
            else:
                fused_features = self.path_fusion(vlm_features, reconstruction_features)
        elif self.use_vlm_path:
            fused_features = vlm_features
        else:
            fused_features = reconstruction_features
        if texts is not None and self.use_vlm_path:
            _, text_features = self.vlm_path(images, texts)
        else:
            text_features = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return fused_features, text_features

    def get_path_importance(self, images):
        reconstruction_features, reconstruction_info = self.reconstruction_path(images)
        if self.path_fusion_strategy == 'adaptive':
            importance = self.path_selector.get_importance(reconstruction_info)
            return importance
        else:
            return {"reconstruction_importance": self.reconstruction_strength}


class VLMFeaturePath(nn.Module):
    def __init__(self, config):
        super().__init__()
        from src.TimeVLM.mae_encoder_plugin import MAEEncoderPlugin
        self.vlm_encoder = MAEEncoderPlugin(config)

    def forward(self, images, texts=None):
        return self.vlm_encoder(images, texts)


class MAEReconstructionPath(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mae_arch = getattr(config, 'mae_arch', 'mae_base')
        self.ckpt_dir = getattr(config, 'mae_ckpt_dir', './ckpt/')
        self._init_mae_model()
        self.reconstruction_extractor = ReconstructionFeatureExtractor(768)

    def _init_mae_model(self):
        mae_archs = {
            'mae_base': mae_vit_base_patch16,
            'mae_large': mae_vit_large_patch16,
            'mae_huge': mae_vit_huge_patch14,
        }
        self.mae_model = mae_archs[self.mae_arch](img_size=224)
        checkpoint_path = os.path.join(self.ckpt_dir, 'mae_visualize_vit_base.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.mae_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.mae_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = self._preprocess(images)
        with torch.no_grad():
            loss, pred, mask = self.mae_model(images, mask_ratio=0.5)
            reconstruction_features = self.reconstruction_extractor(pred, mask)
            reconstruction_info = {
                'reconstruction_loss': loss.item() if isinstance(loss, torch.Tensor) else float(loss),
                'mask_ratio': mask.mean().item(),
                'feature_variance': reconstruction_features.var().item()
            }
        return reconstruction_features, reconstruction_info

    def _preprocess(self, images):
        if isinstance(images, torch.Tensor):
            if images.shape[2] != 224 or images.shape[3] != 224:
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0
                images = F.interpolate(images, size=(224, 224), mode='bicubic')
            elif images.dtype == torch.uint8:
                images = images.float() / 255.0
            return images
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if isinstance(images, list):
            return torch.stack([preprocess(img) for img in images])
        else:
            return preprocess(images).unsqueeze(0)


class ReconstructionFeatureExtractor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            ),
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        ])
        self.feature_aggregator = nn.Sequential(
            nn.Linear(128 + 64, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, pred_patches, mask):
        batch_size = pred_patches.shape[0]
        reconstructed_images = self._patches_to_images(pred_patches)
        features = []
        for extractor in self.feature_extractors:
            feat = extractor(reconstructed_images)
            feat = feat.view(batch_size, -1)
            features.append(feat)
        combined_features = torch.cat(features, dim=-1)
        aggregated_features = self.feature_aggregator(combined_features)
        return aggregated_features

    def _patches_to_images(self, patches):
        B, L, PP = patches.shape
        P = int((PP // 3) ** 0.5)
        H = W = int(L ** 0.5)
        patches = patches.view(B, H, W, P, P, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        images = patches.reshape(B, 3, H * P, W * P)
        return images


class AdaptivePathSelector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, reconstruction_info):
        if isinstance(reconstruction_info, dict):
            quality_score = 1.0 / (1.0 + reconstruction_info.get('reconstruction_loss', 1.0))
        else:
            quality_score = torch.sigmoid(reconstruction_info.var())
        return quality_score

    def get_importance(self, reconstruction_info):
        quality_score = self.forward(reconstruction_info)
        return {"reconstruction_importance": quality_score.item()}


class AdaptivePathFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, vlm_features, reconstruction_features, fusion_weight):
        vlm_weight = 1.0 - fusion_weight
        reconstruction_weight = fusion_weight
        weighted_vlm = vlm_features * vlm_weight
        weighted_reconstruction = reconstruction_features * reconstruction_weight
        combined = torch.cat([weighted_vlm, weighted_reconstruction], dim=-1)
        fused_features = self.fusion_layer(combined)
        return fused_features


class LearnedPathFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, vlm_features, reconstruction_features):
        combined_input = torch.cat([vlm_features, reconstruction_features], dim=-1)
        weights = self.gate(combined_input)
        _ = weights  # keep for potential extension
        fused_features = self.fusion_layer(combined_input)
        return fused_features


class FixedPathFusion(nn.Module):
    def __init__(self, vlm_weight=0.5, reconstruction_weight=0.5):
        super().__init__()
        self.vlm_weight = vlm_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(self, vlm_features, reconstruction_features):
        fused_features = self.vlm_weight * vlm_features + self.reconstruction_weight * reconstruction_features
        return fused_features

