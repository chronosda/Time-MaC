import torch
import torch.nn as nn
import sys
from typing import Optional, Tuple, List
from PIL import Image
import torchvision.transforms as transforms
import os
import einops
import torch.nn.functional as F

# Import MAE models
sys.path.append("../")
from layers.models_mae import *

class MAEReconstructionVLM(nn.Module):
    """
    多模态架构中融合MAE重建能力的创新设计

    核心思想：
    1. 利用MAE重建能力生成高质量的视觉特征
    2. 重建结果作为多模态融合的增强输入
    3. 保持VLM的多模态灵活性，同时获得重建优势
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self._acquire_device()

        # MAE配置
        self.mae_arch = getattr(config, 'mae_arch', 'mae_base')
        self.finetune_type = getattr(config, 'mae_finetune_type', 'ln')
        self.ckpt_dir = getattr(config, 'mae_ckpt_dir', './ckpt/')
        self.load_ckpt = getattr(config, 'mae_load_ckpt', True)

        # 重建导向的参数
        self.reconstruction_ratio = getattr(config, 'reconstruction_ratio', 0.3)  # 30%区域重建
        self.use_reconstruction_features = getattr(config, 'use_reconstruction_features', True)
        self.multimodal_fusion_type = getattr(config, 'multimodal_fusion_type', 'reconstruction_aware')

        # 初始化MAE重建模型
        self._init_mae_reconstruction()

        # 初始化文本处理器
        self._init_text_processor()

        # 初始化重建特征增强器
        self._init_reconstruction_enhancer()

        # 设置特征维度
        self._set_feature_dimensions()

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
        return device

    def _init_mae_reconstruction(self):
        """初始化支持重建的MAE模型"""
        mae_archs = {
            'mae_base': mae_vit_base_patch16,
            'mae_large': mae_vit_large_patch16,
            'mae_huge': mae_vit_huge_patch14,
        }

        if self.mae_arch not in mae_archs:
            raise ValueError(f"Unknown MAE architecture: {self.mae_arch}")

        # 初始化MAE模型
        self.mae_model = mae_archs[self.mae_arch](img_size=224)

        # 加载预训练权重
        if self.load_ckpt:
            checkpoint_path = os.path.join(self.ckpt_dir, 'mae_visualize_vit_base.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.mae_model.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded MAE reconstruction model: {self.mae_arch}")

        # 设置微调策略
        self._setup_finetuning()

        # 计算重建mask
        self._setup_reconstruction_mask()

    def _setup_finetuning(self):
        """设置保守的微调策略"""
        if self.finetune_type == 'ln':
            for name, param in self.mae_model.named_parameters():
                param.requires_grad = 'norm' in name
        elif self.finetune_type == 'none':
            for param in self.mae_model.parameters():
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.mae_model.parameters() if p.requires_grad)
        print(f"MAE reconstruction trainable parameters: {trainable_params:,}")

    def _setup_reconstruction_mask(self):
        """设置重建区域的mask"""
        # 创建可学习的重建mask
        self.register_buffer('reconstruction_mask', self._create_reconstruction_mask())

    def _create_reconstruction_mask(self):
        """创建重建mask - 模仿VisionTS的策略"""
        # 将图像分为输入区域和重建区域
        num_patches = 14 * 14  # 224x224 with patch_size=16
        num_input_patches = int(num_patches * (1 - self.reconstruction_ratio))

        mask = torch.ones(num_patches)
        mask[:num_input_patches] = 0  # 输入区域不遮盖
        mask[num_input_patches:] = 1  # 重建区域遮盖

        return mask.float()

    def _init_text_processor(self):
        """初始化文本处理器"""
        # 简化的文本处理，保持多模态兼容性
        try:
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.text_projection = nn.Linear(768, 768)
        except:
            # 降级处理
            self.tokenizer = None
            self.text_encoder = None
            self.text_projection = nn.Linear(768, 768)

    def _init_reconstruction_enhancer(self):
        """初始化重建特征增强器"""
        if self.use_reconstruction_features:
            # 重建特征增强模块
            self.reconstruction_enhancer = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 768),
                nn.LayerNorm(768)
            )

            # 多模态重建融合模块
            if self.multimodal_fusion_type == 'reconstruction_aware':
                self.multimodal_fusion = ReconstructionAwareFusion(768)
            else:
                self.multimodal_fusion = nn.Sequential(
                    nn.Linear(768 * 2, 768),
                    nn.ReLU(),
                    nn.Linear(768, 768)
                )
        else:
            self.reconstruction_enhancer = nn.Identity()
            self.multimodal_fusion = nn.Identity()

    def _set_feature_dimensions(self):
        """设置特征维度"""
        if self.mae_arch == 'mae_base':
            self.hidden_size = 768
        elif self.mae_arch == 'mae_large':
            self.hidden_size = 1024
        elif self.mae_arch == 'mae_huge':
            self.hidden_size = 1280

        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77
        self.fused_feature_len = self.hidden_size

        # 初始化特征适配器
        self.feature_adapter = nn.Sequential(
            nn.Linear(3, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        ).to(self.device)

    def forward(self, images, texts=None):
        """
        重建导向的多模态前向传播

        Args:
            images: 输入图像
            texts: 可选文本输入

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (重建增强的视觉特征, 文本特征)
        """
        # 1. MAE重建处理
        reconstruction_features, original_features = self._mae_reconstruction_forward(images)

        # 2. 重建特征增强
        enhanced_features = self.reconstruction_enhancer(reconstruction_features)

        # 3. 文本处理
        if texts is not None and self.text_encoder is not None:
            text_features = self._encode_text(texts)
        else:
            batch_size = enhanced_features.shape[0]
            text_features = torch.zeros(batch_size, self.hidden_size).to(self.device)

        # 4. 重建感知的多模态融合
        if self.use_reconstruction_features:
            fused_features = self.multimodal_fusion(enhanced_features, text_features)
        else:
            fused_features = enhanced_features

        return fused_features, text_features

    def _mae_reconstruction_forward(self, images):
        """MAE重建前向传播"""
        # 预处理图像
        images = self._preprocess_images(images)
        images = images.to(self.device)

        with torch.no_grad():
            # 执行MAE重建 - 使用reconstruction_ratio作为mask_ratio
            latent, mask, ids_restore = self.mae_model.forward_encoder(images, self.reconstruction_ratio)
            pred = self.mae_model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

            # 获取原始特征
            original_features = self.mae_model.forward_encoder(images, 0.0)[0]
            original_features = original_features[:, 0, :]  # CLS token

            # 重建特征 - 从重建结果中提取
            reconstructed_patches = self.mae_model.unpatchify(pred)
            reconstruction_features = self._extract_reconstruction_features(reconstructed_patches, mask)

        return reconstruction_features, original_features

    def _extract_reconstruction_features(self, reconstructed_patches, mask):
        """从重建结果中提取特征"""
        batch_size = reconstructed_patches.shape[0]

        # 1. 将mask重塑为图像格式
        # mask shape: [B, num_patches], reconstructed_patches shape: [B, C, H, W]
        num_patches = mask.shape[1]
        patch_size = 16  # MAE patch size
        img_size = 224   # MAE image size
        patches_per_side = img_size // patch_size

        # 2. 重构mask以匹配图像尺寸
        mask_2d = mask.view(batch_size, patches_per_side, patches_per_side)
        mask_4d = mask_2d.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, H_p, W_p, 1, 1]
        mask_4d = mask_4d.expand(-1, -1, -1, -1, patch_size, patch_size)  # [B, 1, H_p, W_p, patch_size, patch_size]
        mask_4d = mask_4d.reshape(batch_size, 1, img_size, img_size)  # [B, 1, H, W]

        # 3. 找到被重建的区域
        reconstructed_regions = reconstructed_patches * mask_4d

        # 4. 从重建区域提取特征
        # 使用简单的平均池化作为重建特征
        reconstruction_features = reconstructed_regions.mean(dim=[2, 3])  # [B, C]

        # 5. 如果需要，可以添加更复杂的特征提取
        if reconstruction_features.shape[1] != self.hidden_size:
            # 使用线性层调整特征维度
            reconstruction_features = self.feature_adapter(reconstruction_features)

        return reconstruction_features

    def _preprocess_images(self, images):
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

        # PIL图像处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if isinstance(images, list):
            return torch.stack([preprocess(img) for img in images])
        else:
            return preprocess(images).unsqueeze(0)

    def _encode_text(self, texts):
        """文本编码"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # 平均池化

        return self.text_projection(text_features)

    def get_reconstruction_loss(self, images):
        """获取重建损失用于辅助训练"""
        images = self._preprocess_images(images)
        images = images.to(self.device)

        # 计算重建损失
        loss, _, _ = self.mae_model(images, mask_ratio=self.reconstruction_ratio)
        return loss


class ReconstructionAwareFusion(nn.Module):
    """重建感知的多模态融合模块"""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 重建质量门控
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 多模态融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, vision_features, text_features):
        # 计算重建质量权重
        quality_weight = self.quality_gate(vision_features)

        # 根据重建质量调整融合权重
        vision_weight = quality_weight
        text_weight = 1.0 - quality_weight

        # 加权融合
        weighted_vision = vision_features * vision_weight
        weighted_text = text_features * text_weight

        # 最终融合
        fused_features = torch.cat([weighted_vision, weighted_text], dim=-1)
        fused_features = self.fusion_layer(fused_features)

        return fused_features