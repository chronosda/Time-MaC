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
    Reconstruction-oriented MAE integration for multimodal pipelines.

    Key ideas:
    1. Use MAE reconstruction to derive high-quality vision features
    2. Feed reconstruction-derived features into multimodal fusion
    3. Keep VLM flexibility while leveraging reconstruction benefits
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self._acquire_device()

        # MAE config
        self.mae_arch = getattr(config, 'mae_arch', 'mae_base')
        self.finetune_type = getattr(config, 'mae_finetune_type', 'ln')
        self.ckpt_dir = getattr(config, 'mae_ckpt_dir', './ckpt/')
        self.load_ckpt = getattr(config, 'mae_load_ckpt', True)

        # Reconstruction-oriented parameters
        self.reconstruction_ratio = getattr(config, 'reconstruction_ratio', 0.3)
        self.use_reconstruction_features = getattr(config, 'use_reconstruction_features', True)
        self.multimodal_fusion_type = getattr(config, 'multimodal_fusion_type', 'reconstruction_aware')

        # Init MAE with reconstruction
        self._init_mae_reconstruction()

        # Init text processor (optional, for compatibility)
        self._init_text_processor()

        # Init reconstruction feature enhancer
        self._init_reconstruction_enhancer()

        # Set feature dims
        self._set_feature_dimensions()

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
        return device

    def _init_mae_reconstruction(self):
        mae_archs = {
            'mae_base': mae_vit_base_patch16,
            'mae_large': mae_vit_large_patch16,
            'mae_huge': mae_vit_huge_patch14,
        }
        if self.mae_arch not in mae_archs:
            raise ValueError(f"Unknown MAE architecture: {self.mae_arch}")

        self.mae_model = mae_archs[self.mae_arch](img_size=224)

        if self.load_ckpt:
            checkpoint_path = os.path.join(self.ckpt_dir, 'mae_visualize_vit_base.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.mae_model.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded MAE reconstruction model: {self.mae_arch}")

        self._setup_finetuning()
        self._setup_reconstruction_mask()

    def _setup_finetuning(self):
        if self.finetune_type == 'ln':
            for name, param in self.mae_model.named_parameters():
                param.requires_grad = 'norm' in name
        elif self.finetune_type == 'none':
            for param in self.mae_model.parameters():
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in self.mae_model.parameters() if p.requires_grad)
        print(f"MAE reconstruction trainable parameters: {trainable_params:,}")

    def _setup_reconstruction_mask(self):
        self.register_buffer('reconstruction_mask', self._create_reconstruction_mask())

    def _create_reconstruction_mask(self):
        num_patches = 14 * 14  # 224x224 with patch_size=16
        num_input_patches = int(num_patches * (1 - self.reconstruction_ratio))
        mask = torch.ones(num_patches)
        mask[:num_input_patches] = 0
        mask[num_input_patches:] = 1
        return mask.float()

    def _init_text_processor(self):
        try:
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.text_projection = nn.Linear(768, 768)
        except Exception:
            self.tokenizer = None
            self.text_encoder = None
            self.text_projection = nn.Linear(768, 768)

    def _init_reconstruction_enhancer(self):
        if self.use_reconstruction_features:
            self.reconstruction_enhancer = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 768),
                nn.LayerNorm(768)
            )
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
        if self.mae_arch == 'mae_base':
            self.hidden_size = 768
        elif self.mae_arch == 'mae_large':
            self.hidden_size = 1024
        elif self.mae_arch == 'mae_huge':
            self.hidden_size = 1280
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77
        self.fused_feature_len = self.hidden_size

        self.feature_adapter = nn.Sequential(
            nn.Linear(3, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        ).to(self.device)

    def forward(self, images, texts=None):
        reconstruction_features, original_features = self._mae_reconstruction_forward(images)

        enhanced_features = self.reconstruction_enhancer(reconstruction_features)

        if texts is not None and self.text_encoder is not None:
            text_features = self._encode_text(texts)
        else:
            batch_size = enhanced_features.shape[0]
            text_features = torch.zeros(batch_size, self.hidden_size).to(self.device)

        if self.use_reconstruction_features:
            fused_features = self.multimodal_fusion(enhanced_features, text_features)
        else:
            fused_features = enhanced_features

        return fused_features, text_features

    def _mae_reconstruction_forward(self, images):
        images = self._preprocess_images(images)
        images = images.to(self.device)

        with torch.no_grad():
            latent, mask, ids_restore = self.mae_model.forward_encoder(images, self.reconstruction_ratio)
            pred = self.mae_model.forward_decoder(latent, ids_restore)

            original_features = self.mae_model.forward_encoder(images, 0.0)[0]
            original_features = original_features[:, 0, :]

            reconstructed_patches = self.mae_model.unpatchify(pred)
            reconstruction_features = self._extract_reconstruction_features(reconstructed_patches, mask)

        return reconstruction_features, original_features

    def _extract_reconstruction_features(self, reconstructed_patches, mask):
        batch_size = reconstructed_patches.shape[0]
        num_patches = mask.shape[1]
        patch_size = 16
        img_size = 224
        patches_per_side = img_size // patch_size

        mask_2d = mask.view(batch_size, patches_per_side, patches_per_side)
        mask_4d = mask_2d.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        mask_4d = mask_4d.expand(-1, -1, -1, -1, patch_size, patch_size)
        mask_4d = mask_4d.reshape(batch_size, 1, img_size, img_size)

        reconstructed_regions = reconstructed_patches * mask_4d
        reconstruction_features = reconstructed_regions.mean(dim=[2, 3])

        if reconstruction_features.shape[1] != self.hidden_size:
            reconstruction_features = self.feature_adapter(reconstruction_features)

        return reconstruction_features

    def _preprocess_images(self, images):
        if isinstance(images, torch.Tensor):
            if images.shape[2] != 224 or images.shape[3] != 224:
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0
                images = F.interpolate(images, size=(224, 224), mode='bicubic')
            elif images.dtype == torch.uint8:
                images = images.float() / 255.0
            return images

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
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        return self.text_projection(text_features)

    def get_reconstruction_loss(self, images):
        images = self._preprocess_images(images)
        images = images.to(self.device)
        loss, _, _ = self.mae_model(images, mask_ratio=self.reconstruction_ratio)
        return loss


class ReconstructionAwareFusion(nn.Module):
    """Reconstruction-aware multimodal fusion module"""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, vision_features, text_features):
        quality_weight = self.quality_gate(vision_features)
        vision_weight = quality_weight
        text_weight = 1.0 - quality_weight
        weighted_vision = vision_features * vision_weight
        weighted_text = text_features * text_weight
        fused_features = torch.cat([weighted_vision, weighted_text], dim=-1)
        fused_features = self.fusion_layer(fused_features)
        return fused_features

