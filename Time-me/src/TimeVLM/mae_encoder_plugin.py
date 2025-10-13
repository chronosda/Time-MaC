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

class MAEEncoderPlugin(nn.Module):
    """
    Pre-trained MAE encoder plugin for TimeVLM image module.
    This plugin provides vision encoding capabilities using Masked Autoencoder (MAE)
    as a drop-in replacement for VLM encoders.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self._acquire_device()

        # MAE architecture configuration
        self.mae_arch = getattr(config, 'mae_arch', 'mae_base')
        self.finetune_type = getattr(config, 'mae_finetune_type', 'ln')
        self.ckpt_dir = getattr(config, 'mae_ckpt_dir', './ckpt/')
        self.load_ckpt = getattr(config, 'mae_load_ckpt', True)

        # Initialize MAE encoder
        self._init_mae_encoder()

        # Initialize text processor (fallback for multimodal compatibility)
        self._init_text_processor()

        # Set feature dimensions based on MAE architecture
        self._set_feature_dimensions()

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_mae_encoder(self):
        """Initialize the MAE encoder based on architecture selection."""
        mae_archs = {
            'mae_base': mae_vit_base_patch16,
            'mae_large': mae_vit_large_patch16,
            'mae_huge': mae_vit_huge_patch14
        }

        if self.mae_arch not in mae_archs:
            raise ValueError(f"Unknown MAE architecture: {self.mae_arch}. "
                           f"Available: {list(mae_archs.keys())}")

        print(f"Initializing MAE encoder: {self.mae_arch}")
        self.mae_encoder = mae_archs[self.mae_arch]()

        # Load pre-trained weights if specified
        if self.load_ckpt:
            self._load_mae_checkpoint()

        # Move to device
        self.mae_encoder.to(self.device)

        # Set requires_grad based on finetune_type
        self._set_finetuning_mode()

        # Set evaluation mode (no decoder needed for encoding)
        self.mae_encoder.eval()
        for param in self.mae_encoder.parameters():
            param.requires_grad = False

    def _load_mae_checkpoint(self):
        """Load pre-trained MAE checkpoint."""
        mae_ckpt_files = {
            'mae_base': 'mae_visualize_vit_base.pth',
            'mae_large': 'mae_visualize_vit_large.pth',
            'mae_huge': 'mae_visualize_vit_huge.pth'
        }

        ckpt_file = mae_ckpt_files[self.mae_arch]
        ckpt_path = f"{self.ckpt_dir}{ckpt_file}"

        try:
            if not os.path.exists(ckpt_path):
                print(f"MAE checkpoint not found at {ckpt_path}")
                print("Please download the checkpoint from:")
                print("https://dl.fbaipublicfiles.com/mae/visualize/")
                return

            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.mae_encoder.load_state_dict(checkpoint['model'], strict=True)
            print(f"Successfully loaded MAE checkpoint: {ckpt_file}")
        except Exception as e:
            print(f"Failed to load MAE checkpoint: {e}")

    def _set_finetuning_mode(self):
        """Set finetuning mode for MAE encoder parameters."""
        if self.finetune_type == 'full':
            for param in self.mae_encoder.parameters():
                param.requires_grad = True
        elif self.finetune_type == 'none':
            for param in self.mae_encoder.parameters():
                param.requires_grad = False
        elif self.finetune_type == 'ln':
            for name, param in self.mae_encoder.named_parameters():
                param.requires_grad = 'norm' in name
        elif self.finetune_type == 'bias':
            for name, param in self.mae_encoder.named_parameters():
                param.requires_grad = 'bias' in name
        elif self.finetune_type == 'mlp':
            for name, param in self.mae_encoder.named_parameters():
                param.requires_grad = '.mlp.' in name
        elif self.finetune_type == 'attn':
            for name, param in self.mae_encoder.named_parameters():
                param.requires_grad = '.attn.' in name

        trainable_params = sum(p.numel() for p in self.mae_encoder.parameters() if p.requires_grad)
        print(f"MAE encoder trainable parameters: {trainable_params:,}")

    def _init_text_processor(self):
        """Initialize text processor for multimodal compatibility."""
        # For MAE encoder plugin, we use a simple tokenizer for compatibility
        try:
            from transformers import BertTokenizer
            self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        except ImportError:
            print("Warning: transformers not available, using dummy text processing")
            self.text_tokenizer = None

    def _set_feature_dimensions(self):
        """Set feature dimensions based on MAE architecture."""
        embed_dims = {
            'mae_base': 768,
            'mae_large': 1024,
            'mae_huge': 1280
        }

        self.hidden_size = embed_dims[self.mae_arch]
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77  # Standard for compatibility
        self.fused_feature_len = 196  # 14x14 patches for base model

    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess images for MAE encoder.

        Args:
            images: List of PIL Image objects or tensor

        Returns:
            torch.Tensor: Preprocessed images [B, 3, 224, 224]
        """
        if isinstance(images, torch.Tensor):
            # Resize tensor images to 224x224
            if images.shape[2] != 224 or images.shape[3] != 224:
                # Convert to float if needed
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0

                images = torch.nn.functional.interpolate(
                    images,
                    size=(224, 224),
                    mode='bicubic',
                    align_corners=False
                )
            return images

        # Standard MAE preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if isinstance(images, list):
            # List of PIL images
            processed_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    processed_images.append(preprocess(img))
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
            return torch.stack(processed_images)
        elif isinstance(images, Image.Image):
            # Single PIL image
            return preprocess(images).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported image input type: {type(images)}")

    def extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract vision features using MAE encoder.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            torch.Tensor: Vision features [B, hidden_size]
        """
        with torch.no_grad():
            # Extract patch embeddings using MAE encoder
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

            # Use CLS token as the overall image representation
            vision_features = x[:, 0, :]  # [B, hidden_size]

        return vision_features

    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features using MAE encoder.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            torch.Tensor: Patch features [B, num_patches, hidden_size]
        """
        with torch.no_grad():
            # Extract patch embeddings using MAE encoder
            x = self.mae_encoder.patch_embed(images)

            # Add position embeddings
            x = x + self.mae_encoder.pos_embed[:, 1:, :]

            # Apply Transformer blocks (without CLS token)
            for blk in self.mae_encoder.blocks:
                x = blk(x)

            # Apply normalization
            x = self.mae_encoder.norm(x)

            # Return patch features (excluding CLS token)
            patch_features = x  # [B, num_patches, hidden_size]

        return patch_features

    def encode_images(self, images) -> torch.Tensor:
        """
        Encode images using MAE encoder (main interface).

        Args:
            images: Input images (PIL Image, list of PIL Images, or torch.Tensor)

        Returns:
            torch.Tensor: Encoded image features [B, hidden_size]
        """
        # Preprocess images
        images = self.preprocess_images(images)

        # Move to device
        images = images.to(self.device)

        # Extract vision features
        vision_features = self.extract_vision_features(images)

        return vision_features

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text using compatible text processor (for multimodal compatibility).

        Args:
            texts: List of text strings

        Returns:
            torch.Tensor: Text features [B, hidden_size]
        """
        if self.text_tokenizer is None:
            # Return dummy features if no tokenizer available
            batch_size = len(texts) if isinstance(texts, list) else 1
            return torch.zeros(batch_size, self.hidden_size).to(self.device)

        # Tokenize text
        encoded = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_input_text_length,
            return_tensors="pt"
        ).to(self.device)

        # Create dummy text features (since MAE doesn't have text encoder)
        # In practice, you might want to integrate with a proper text encoder
        batch_size = len(texts) if isinstance(texts, list) else 1
        text_features = torch.randn(batch_size, self.hidden_size).to(self.device)

        return text_features

    def forward(self, images, texts=None):
        """
        Forward pass for multimodal encoding.

        Args:
            images: Input images
            texts: Optional text inputs

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (vision_features, text_features)
        """
        # Encode images
        vision_features = self.encode_images(images)

        # Encode texts if provided
        if texts is not None:
            text_features = self.encode_text(texts)
        else:
            # Return dummy text features for compatibility
            batch_size = vision_features.shape[0]
            text_features = torch.zeros(batch_size, self.hidden_size).to(self.device)

        return vision_features, text_features

    def get_feature_extractor(self):
        """Get the underlying MAE encoder for advanced usage."""
        return self.mae_encoder

    def get_config(self):
        """Get the current configuration."""
        return {
            'mae_arch': self.mae_arch,
            'finetune_type': self.finetune_type,
            'hidden_size': self.hidden_size,
            'device': str(self.device),
            'load_ckpt': self.load_ckpt
        }


class MAEEncoderWithDecoder(MAEEncoderPlugin):
    """
    Extended MAE encoder plugin that includes decoder capabilities
    for image reconstruction tasks.
    """

    def __init__(self, config):
        super().__init__(config)

        # Enable decoder components
        self.decoder_enabled = getattr(config, 'mae_decoder_enabled', True)

        if self.decoder_enabled:
            # Enable decoder components
            for param in self.mae_encoder.decoder_embed.parameters():
                param.requires_grad = False
            for param in self.mae_encoder.decoder_blocks.parameters():
                param.requires_grad = False
            for param in self.mae_encoder.decoder_norm.parameters():
                param.requires_grad = False
            for param in self.mae_encoder.decoder_pred.parameters():
                param.requires_grad = False

    def reconstruct_images(self, images: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct images using MAE decoder.

        Args:
            images: Input images [B, 3, H, W]
            mask_ratio: Ratio of patches to mask

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (reconstructed_images, mask)
        """
        with torch.no_grad():
            # Forward pass through MAE with decoder
            _, pred, mask = self.mae_encoder(images, mask_ratio=mask_ratio)

            # Reconstruct images
            reconstructed_images = self.mae_encoder.unpatchify(pred)

        return reconstructed_images, mask

    def encode_with_reconstruction(self, images: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features and reconstruct images simultaneously.

        Args:
            images: Input images [B, 3, H, W]
            mask_ratio: Ratio of patches to mask

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (features, reconstructed_images, mask)
        """
        # Extract features
        features = self.extract_vision_features(images)

        # Reconstruct images
        reconstructed_images, mask = self.reconstruct_images(images, mask_ratio)

        return features, reconstructed_images, mask