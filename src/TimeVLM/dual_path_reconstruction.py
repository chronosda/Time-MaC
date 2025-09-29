import torch
import torch.nn as nn
import sys
from typing import Optional, Tuple, List
import os
import einops
import torch.nn.functional as F

# Import MAE models
sys.path.append("../")
from layers.models_mae import *

class DualPathReconstructionVLM(nn.Module):
    """
    双路径重建增强VLM架构

    核心设计：
    路径1: 传统VLM特征提取路径
    路径2: MAE重建能力路径
    智能融合：根据任务需求动态结合两条路径
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self._acquire_device()

        # 路径配置
        self.use_vlm_path = getattr(config, 'use_vlm_path', True)
        self.use_reconstruction_path = getattr(config, 'use_reconstruction_path', True)
        self.path_fusion_strategy = getattr(config, 'path_fusion_strategy', 'adaptive')  # adaptive, fixed, learned

        # MAE重建配置
        self.mae_arch = getattr(config, 'mae_arch', 'mae_base')
        self.reconstruction_strength = getattr(config, 'reconstruction_strength', 0.7)

        # 初始化双路径
        self._init_dual_paths()

        # 初始化融合模块
        self._init_path_fusion()

        # 设置维度
        self._set_dimensions()

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
        return device

    def _init_dual_paths(self):
        """初始化双路径架构"""
        # 路径1: 传统VLM特征提取
        if self.use_vlm_path:
            self.vlm_path = VLMFeaturePath(self.config)

        # 路径2: MAE重建路径
        if self.use_reconstruction_path:
            self.reconstruction_path = MAEReconstructionPath(self.config)

    def _init_path_fusion(self):
        """初始化路径融合模块"""
        if not (self.use_vlm_path and self.use_reconstruction_path):
            return

        hidden_size = 768  # MAE-base

        if self.path_fusion_strategy == 'adaptive':
            # 自适应融合 - 根据输入特性动态调整
            self.path_selector = AdaptivePathSelector(hidden_size)
            self.path_fusion = AdaptivePathFusion(hidden_size)
        elif self.path_fusion_strategy == 'learned':
            # 学习融合 - 可训练的融合权重
            self.path_fusion = LearnedPathFusion(hidden_size)
        else:  # fixed
            # 固定融合 - 预设权重
            self.path_fusion = FixedPathFusion(
                vlm_weight=1.0 - self.reconstruction_strength,
                reconstruction_weight=self.reconstruction_strength
            )

    def _set_dimensions(self):
        """设置特征维度"""
        self.hidden_size = 768
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77
        self.fused_feature_len = self.hidden_size

    def forward(self, images, texts=None):
        """
        双路径前向传播
        """
        batch_size = images.shape[0] if isinstance(images, torch.Tensor) else len(images)

        # 路径1: VLM特征提取
        vlm_features = torch.zeros(batch_size, self.hidden_size).to(self.device)
        if self.use_vlm_path:
            vlm_features, _ = self.vlm_path(images, texts)

        # 路径2: MAE重建特征
        reconstruction_features = torch.zeros(batch_size, self.hidden_size).to(self.device)
        if self.use_reconstruction_path:
            reconstruction_features, reconstruction_info = self.reconstruction_path(images)

        # 路径融合
        if self.use_vlm_path and self.use_reconstruction_path:
            if self.path_fusion_strategy == 'adaptive':
                # 根据重建质量自适应选择
                fusion_weight = self.path_selector(reconstruction_info)
                fused_features = self.path_fusion(vlm_features, reconstruction_features, fusion_weight)
            else:
                fused_features = self.path_fusion(vlm_features, reconstruction_features)
        elif self.use_vlm_path:
            fused_features = vlm_features
        else:
            fused_features = reconstruction_features

        # 文本特征（保持兼容性）
        if texts is not None and self.use_vlm_path:
            _, text_features = self.vlm_path(images, texts)
        else:
            text_features = torch.zeros(batch_size, self.hidden_size).to(self.device)

        return fused_features, text_features

    def get_path_importance(self, images):
        """获取路径重要性分析"""
        if not self.use_reconstruction_path:
            return {"reconstruction_importance": 0.0}

        reconstruction_features, reconstruction_info = self.reconstruction_path(images)
        if self.path_fusion_strategy == 'adaptive':
            importance = self.path_selector.get_importance(reconstruction_info)
            return importance
        else:
            return {"reconstruction_importance": self.reconstruction_strength}


class VLMFeaturePath(nn.Module):
    """传统VLM特征提取路径"""

    def __init__(self, config):
        super().__init__()
        # 这里可以复用原有的VLM逻辑
        from src.TimeVLM.mae_encoder_plugin import MAEEncoderPlugin
        self.vlm_encoder = MAEEncoderPlugin(config)

    def forward(self, images, texts=None):
        return self.vlm_encoder(images, texts)


class MAEReconstructionPath(nn.Module):
    """MAE重建特征提取路径"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mae_arch = getattr(config, 'mae_arch', 'mae_base')
        self.ckpt_dir = getattr(config, 'mae_ckpt_dir', './ckpt/')

        # 初始化MAE模型
        self._init_mae_model()

        # 重建特征提取器
        self.reconstruction_extractor = ReconstructionFeatureExtractor(768)

    def _init_mae_model(self):
        """初始化MAE模型"""
        mae_archs = {
            'mae_base': mae_vit_base_patch16,
            'mae_large': mae_vit_large_patch16,
            'mae_huge': mae_vit_huge_patch14,
        }

        self.mae_model = mae_archs[self.mae_arch](img_size=224)

        # 加载权重
        checkpoint_path = os.path.join(self.ckpt_dir, 'mae_visualize_vit_base.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.mae_model.load_state_dict(checkpoint['model'], strict=False)

        # 冻结参数用于推理
        for param in self.mae_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        """重建路径前向传播"""
        # 预处理
        images = self._preprocess(images)

        with torch.no_grad():
            # 执行重建
            loss, pred, mask = self.mae_model(images, mask_ratio=0.5)

            # 提取重建特征
            reconstruction_features = self.reconstruction_extractor(pred, mask)
            reconstruction_info = {
                'reconstruction_loss': loss.item(),
                'mask_ratio': mask.mean().item(),
                'feature_variance': reconstruction_features.var().item()
            }

        return reconstruction_features, reconstruction_info

    def _preprocess(self, images):
        """图像预处理"""
        if isinstance(images, torch.Tensor):
            if images.shape[2] != 224 or images.shape[3] != 224:
                # Convert to float before interpolation
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0
                images = F.interpolate(images, size=(224, 224), mode='bicubic')
            elif images.dtype == torch.uint8:
                images = images.float() / 255.0
            return images

        # PIL处理
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
    """从MAE重建结果中提取特征的模块"""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 多尺度特征提取
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

        # 特征聚合
        self.feature_aggregator = nn.Sequential(
            nn.Linear(128 + 64, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, pred_patches, mask):
        """从预测的patch中提取特征"""
        batch_size = pred_patches.shape[0]

        # 重构为图像格式
        reconstructed_images = self._patches_to_images(pred_patches)

        # 多尺度特征提取
        features = []
        for extractor in self.feature_extractors:
            feat = extractor(reconstructed_images)
            feat = feat.view(batch_size, -1)
            features.append(feat)

        # 聚合特征
        combined_features = torch.cat(features, dim=-1)
        aggregated_features = self.feature_aggregator(combined_features)

        return aggregated_features

    def _patches_to_images(self, patches):
        """将patches转换为图像格式"""
        # patches: [B, L, P*P*3] -> [B, 3, H, W]
        B, L, PP = patches.shape
        P = int((PP // 3) ** 0.5)  # patch size
        H = W = int(L ** 0.5)  # image size in patches

        patches = patches.view(B, H, W, P, P, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # [B, 3, H, P, W, P]
        images = patches.reshape(B, 3, H * P, W * P)

        return images


class AdaptivePathSelector(nn.Module):
    """自适应路径选择器"""

    def __init__(self, hidden_size):
        super().__init__()
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, reconstruction_info):
        """根据重建信息选择路径权重"""
        # 这里可以基于重建损失、mask比例等信息动态调整
        # 简化实现：基于重建特征的方差判断重建质量
        if isinstance(reconstruction_info, dict):
            quality_score = 1.0 / (1.0 + reconstruction_info.get('reconstruction_loss', 1.0))
        else:
            # 基于特征方差
            quality_score = torch.sigmoid(reconstruction_info.var())

        return quality_score

    def get_importance(self, reconstruction_info):
        """获取路径重要性"""
        quality_score = self.forward(reconstruction_info)
        return {"reconstruction_importance": quality_score.item()}


class AdaptivePathFusion(nn.Module):
    """自适应路径融合"""

    def __init__(self, hidden_size):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, vlm_features, reconstruction_features, fusion_weight):
        """基于权重的特征融合"""
        # 动态调整权重
        vlm_weight = 1.0 - fusion_weight
        reconstruction_weight = fusion_weight

        # 加权融合
        weighted_vlm = vlm_features * vlm_weight
        weighted_reconstruction = reconstruction_features * reconstruction_weight

        # 最终融合
        combined = torch.cat([weighted_vlm, weighted_reconstruction], dim=-1)
        fused_features = self.fusion_layer(combined)

        return fused_features


class LearnedPathFusion(nn.Module):
    """学习型路径融合"""

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
        """学习权重融合"""
        # 计算融合权重
        combined_input = torch.cat([vlm_features, reconstruction_features], dim=-1)
        weights = self.gate(combined_input)

        # 加权融合
        weighted_features = weights[:, 0:1] * vlm_features + weights[:, 1:2] * reconstruction_features

        # 最终融合
        fused_features = self.fusion_layer(combined_input)

        return fused_features


class FixedPathFusion(nn.Module):
    """固定权重路径融合"""

    def __init__(self, vlm_weight=0.5, reconstruction_weight=0.5):
        super().__init__()
        self.vlm_weight = vlm_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(self, vlm_features, reconstruction_features):
        """固定权重融合"""
        fused_features = self.vlm_weight * vlm_features + self.reconstruction_weight * reconstruction_features
        return fused_features