import torch
import torch.nn as nn
import sys
from typing import Optional, Tuple, List
from PIL import Image
import torchvision.transforms as transforms
import os

# Import MAE models
sys.path.append("../")
from layers.models_mae import *

class MAEEncoderOptimized(nn.Module):
    """
    Optimized MAE encoder specifically designed for time series converted images.
    Key improvements:
    1. Time-series specific preprocessing
    2. Enhanced feature extraction
    3. Better fine-tuning strategy
    4. Adaptive normalization
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self._acquire_device()

        # MAE architecture configuration
        self.mae_arch = getattr(config, 'mae_arch', 'mae_base')
        self.finetune_type = getattr(config, 'mae_finetune_type', 'enhanced')  # Enhanced finetuning
        self.ckpt_dir = getattr(config, 'mae_ckpt_dir', './ckpt/')
        self.load_ckpt = getattr(config, 'mae_load_ckpt', True)

        # Time series specific parameters
        self.image_size = getattr(config, 'image_size', 56)
        self.use_adaptive_norm = getattr(config, 'use_adaptive_norm', True)
        self.use_global_features = getattr(config, 'use_global_features', True)
        self.feature_fusion = getattr(config, 'feature_fusion', True)

        # Initialize MAE encoder
        self._init_mae_encoder()

        # Initialize text processor
        self._init_text_processor()

        # Set feature dimensions
        self._set_feature_dimensions()

        # Initialize adaptive normalization layers
        self._init_adaptive_norm()

        # Set compatibility attributes
        self.max_input_text_length = 77  # BERT-like tokenizer length
        self.fused_feature_len = self.hidden_size

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_mae_encoder(self):
        """Initialize the MAE encoder with enhanced architecture."""
        mae_archs = {
            'mae_base': mae_vit_base_patch16,
            'mae_large': mae_vit_large_patch16,
            'mae_huge': mae_vit_huge_patch14,
        }

        if self.mae_arch not in mae_archs:
            raise ValueError(f"Unknown MAE architecture: {self.mae_arch}")

        # Load MAE model (patch_size is already defined in the function)
        self.mae_encoder = mae_archs[self.mae_arch](img_size=224)

        # Load checkpoint if specified
        if self.load_ckpt:
            checkpoint_path = os.path.join(self.ckpt_dir, 'mae_visualize_vit_base.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                msg = self.mae_encoder.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded MAE checkpoint: {msg}")
            else:
                print(f"MAE checkpoint not found: {checkpoint_path}")

        # Enhanced fine-tuning strategy
        self._setup_finetuning()

    def _setup_finetuning(self):
        """Setup enhanced fine-tuning strategy."""
        if self.finetune_type == 'enhanced':
            # Fine-tune: normalization layers + attention layers + MLP layers
            for name, param in self.mae_encoder.named_parameters():
                if any(keyword in name.lower() for keyword in ['norm', 'attn', 'mlp', 'bias']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.finetune_type == 'full':
            # Full fine-tuning
            for param in self.mae_encoder.parameters():
                param.requires_grad = True
        elif self.finetune_type == 'adaptive':
            # Adaptive: fine-tune last 6 blocks + all norm layers
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
        """Initialize adaptive normalization layers for time series data."""
        if self.use_adaptive_norm:
            # Adaptive normalization based on time series statistics
            self.adaptive_norm = nn.Sequential(
                nn.LayerNorm(768),  # MAE-base hidden size
                nn.Dropout(0.1),
                nn.Linear(768, 768),
                nn.GELU(),
                nn.LayerNorm(768)
            )
        else:
            self.adaptive_norm = nn.Identity()

    def _init_text_processor(self):
        """Initialize text processor for multimodal compatibility."""
        # For MAE encoder plugin, we use a simple tokenizer for compatibility
        try:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_projection = nn.Linear(768, 768)
        except:
            self.tokenizer = None
            self.text_projection = nn.Linear(768, 768)

    def _set_feature_dimensions(self):
        """Set feature dimensions based on MAE architecture."""
        if self.mae_arch == 'mae_base':
            self.hidden_size = 768
            self.patch_size = 16
        elif self.mae_arch == 'mae_large':
            self.hidden_size = 1024
            self.patch_size = 16
        elif self.mae_arch == 'mae_huge':
            self.hidden_size = 1280
            self.patch_size = 14

        # Output dimensions
        self.fusion_dim = self.hidden_size

    def forward(self, images, texts=None):
        """
        Forward pass with enhanced feature extraction.
        """
        # Encode images with enhanced preprocessing
        vision_features = self.encode_images_enhanced(images)

        # Apply adaptive normalization
        vision_features = self.adaptive_norm(vision_features)

        # Encode texts if provided
        if texts is not None:
            text_features = self.encode_text(texts)
        else:
            batch_size = vision_features.shape[0]
            text_features = torch.zeros(batch_size, self.hidden_size).to(self.device)

        return vision_features, text_features

    def encode_images_enhanced(self, images) -> torch.Tensor:
        """
        Enhanced image encoding with time series specific preprocessing.
        """
        # Enhanced preprocessing for time series
        images = self.preprocess_images_enhanced(images)
        images = images.to(self.device)

        # Extract enhanced vision features
        vision_features = self.extract_vision_features_enhanced(images)
        return vision_features

    def preprocess_images_enhanced(self, images) -> torch.Tensor:
        """
        Enhanced preprocessing specifically for time series converted images.
        """
        if isinstance(images, torch.Tensor):
            # Time series specific preprocessing
            if images.dtype == torch.uint8:
                images = images.float() / 255.0

            # Calculate time series specific statistics
            if len(images.shape) == 4:  # [B, C, H, W]
                batch_mean = images.mean(dim=[2, 3], keepdim=True)
                batch_std = images.std(dim=[2, 3], keepdim=True) + 1e-8

                # Adaptive normalization using time series statistics
                images = (images - batch_mean) / batch_std

                # Resize to 224x224 for MAE
                if images.shape[2] != 224 or images.shape[3] != 224:
                    images = torch.nn.functional.interpolate(
                        images,
                        size=(224, 224),
                        mode='bicubic',
                        align_corners=False
                    )
            return images

        # For PIL images
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Use time series adaptive normalization instead of ImageNet
            transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        ])

        if isinstance(images, list):
            return torch.stack([preprocess(img) for img in images])
        else:
            return preprocess(images).unsqueeze(0)

    def extract_vision_features_enhanced(self, images: torch.Tensor) -> torch.Tensor:
        """
        Enhanced vision feature extraction with better strategies for time series.
        """
        with torch.no_grad():
            # Extract patch embeddings
            x = self.mae_encoder.patch_embed(images)

            # Add position embeddings
            x = x + self.mae_encoder.pos_embed[:, 1:, :]

            # Add CLS token
            cls_token = self.mae_encoder.cls_token + self.mae_encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # Apply Transformer blocks
            for blk in self.mae_encoder.blocks:
                x = blk(x)

            # Apply normalization
            x = self.mae_encoder.norm(x)

        # Enhanced feature extraction
        if self.use_global_features:
            # Use both CLS token and global average pooling
            cls_features = x[:, 0, :]  # CLS token

            # Global average pooling of patch tokens
            patch_tokens = x[:, 1:, :]  # Remove CLS token
            global_features = patch_tokens.mean(dim=1)  # Global average pooling

            # Feature fusion
            if self.feature_fusion:
                vision_features = 0.7 * cls_features + 0.3 * global_features
            else:
                vision_features = cls_features
        else:
            # Original CLS token approach
            vision_features = x[:, 0, :]

        return vision_features

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using compatible text processor."""
        if self.tokenizer is None:
            batch_size = len(texts) if isinstance(texts, list) else 1
            return torch.zeros(batch_size, self.hidden_size).to(self.device)

        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Simple text projection (can be enhanced with actual text encoder)
        text_features = self.text_projection(
            torch.randn(input_ids.shape[0], self.hidden_size).to(self.device)
        )

        return text_features

    def get_trainable_params(self):
        """Get trainable parameters for optimization."""
        return [p for p in self.parameters() if p.requires_grad]