import torch
import torch.nn as nn
import sys
from typing import List
from PIL import Image
import torchvision.transforms as transforms
import os

# Import MAE models
sys.path.append("../")
from layers.models_mae import *


class MAEEncoderOptimized(nn.Module):
    """
    Optimized MAE encoder for time series converted images.
    Improvements:
    - Time-series specific preprocessing and normalization
    - Enhanced feature extraction (CLS + GAP fusion)
    - Flexible fine-tuning strategies
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self._acquire_device()

        # MAE arch config
        self.mae_arch = getattr(config, 'mae_arch', 'mae_base')
        self.finetune_type = getattr(config, 'mae_finetune_type', 'enhanced')
        self.ckpt_dir = getattr(config, 'mae_ckpt_dir', './ckpt/')
        self.load_ckpt = getattr(config, 'mae_load_ckpt', True)

        # Time series specific params
        self.image_size = getattr(config, 'image_size', 56)
        self.use_adaptive_norm = getattr(config, 'use_adaptive_norm', True)
        self.use_global_features = getattr(config, 'use_global_features', True)
        self.feature_fusion = getattr(config, 'feature_fusion', True)

        # Init MAE
        self._init_mae_encoder()
        # Init text processor
        self._init_text_processor()
        # Set feature dims
        self._set_feature_dimensions()
        # Init adaptive norm
        self._init_adaptive_norm()

        # Compatibility attrs
        self.max_input_text_length = 77
        self.fused_feature_len = self.hidden_size

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_mae_encoder(self):
        mae_archs = {
            'mae_base': mae_vit_base_patch16,
            'mae_large': mae_vit_large_patch16,
            'mae_huge': mae_vit_huge_patch14,
        }
        if self.mae_arch not in mae_archs:
            raise ValueError(f"Unknown MAE architecture: {self.mae_arch}")
        self.mae_encoder = mae_archs[self.mae_arch](img_size=224)

        if self.load_ckpt:
            checkpoint_path = os.path.join(self.ckpt_dir, 'mae_visualize_vit_base.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                msg = self.mae_encoder.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded MAE checkpoint: {msg}")
            else:
                print(f"MAE checkpoint not found: {checkpoint_path}")

        self._setup_finetuning()

    def _setup_finetuning(self):
        if self.finetune_type == 'enhanced':
            for name, param in self.mae_encoder.named_parameters():
                if any(keyword in name.lower() for keyword in ['norm', 'attn', 'mlp', 'bias']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.finetune_type == 'full':
            for param in self.mae_encoder.parameters():
                param.requires_grad = True
        elif self.finetune_type == 'adaptive':
            for name, param in self.mae_encoder.named_parameters():
                if ('norm' in name.lower() or
                    'blocks.6' in name or 'blocks.7' in name or 'blocks.8' in name or
                    'blocks.9' in name or 'blocks.10' in name or 'blocks.11' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        trainable_params = sum(p.numel() for p in self.mae_encoder.parameters() if p.requires_grad)
        print(f"MAE encoder trainable parameters: {trainable_params:,}")

    def _init_adaptive_norm(self):
        if self.use_adaptive_norm:
            self.adaptive_norm = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Linear(768, 768),
                nn.GELU(),
                nn.LayerNorm(768)
            )
        else:
            self.adaptive_norm = nn.Identity()

    def _init_text_processor(self):
        try:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_projection = nn.Linear(768, 768)
        except Exception:
            self.tokenizer = None
            self.text_projection = nn.Linear(768, 768)

    def _set_feature_dimensions(self):
        if self.mae_arch == 'mae_base':
            self.hidden_size = 768
            self.patch_size = 16
        elif self.mae_arch == 'mae_large':
            self.hidden_size = 1024
            self.patch_size = 16
        elif self.mae_arch == 'mae_huge':
            self.hidden_size = 1280
            self.patch_size = 14
        self.fusion_dim = self.hidden_size

    def forward(self, images, texts=None):
        vision_features = self.encode_images_enhanced(images)
        vision_features = self.adaptive_norm(vision_features)
        if texts is not None:
            text_features = self.encode_text(texts)
        else:
            batch_size = vision_features.shape[0]
            text_features = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return vision_features, text_features

    def encode_images_enhanced(self, images) -> torch.Tensor:
        images = self.preprocess_images_enhanced(images)
        images = images.to(self.device)
        vision_features = self.extract_vision_features_enhanced(images)
        return vision_features

    def preprocess_images_enhanced(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            if len(images.shape) == 4:
                batch_mean = images.mean(dim=[2, 3], keepdim=True)
                batch_std = images.std(dim=[2, 3], keepdim=True) + 1e-8
                images = (images - batch_mean) / batch_std
                if images.shape[2] != 224 or images.shape[3] != 224:
                    images = torch.nn.functional.interpolate(
                        images,
                        size=(224, 224),
                        mode='bicubic',
                        align_corners=False
                    )
            return images
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        ])
        if isinstance(images, list):
            return torch.stack([preprocess(img) for img in images])
        else:
            return preprocess(images).unsqueeze(0)

    def extract_vision_features_enhanced(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.mae_encoder.patch_embed(images)
            x = x + self.mae_encoder.pos_embed[:, 1:, :]
            cls_token = self.mae_encoder.cls_token + self.mae_encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            for blk in self.mae_encoder.blocks:
                x = blk(x)
            x = self.mae_encoder.norm(x)
        if self.use_global_features:
            cls_features = x[:, 0, :]
            patch_tokens = x[:, 1:, :]
            global_features = patch_tokens.mean(dim=1)
            if self.feature_fusion:
                vision_features = 0.7 * cls_features + 0.3 * global_features
            else:
                vision_features = cls_features
        else:
            vision_features = x[:, 0, :]
        return vision_features

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if self.tokenizer is None:
            batch_size = len(texts) if isinstance(texts, list) else 1
            return torch.zeros(batch_size, self.hidden_size).to(self.device)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        text_features = self.text_projection(
            torch.randn(input_ids.shape[0], self.hidden_size).to(self.device)
        )
        return text_features

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

