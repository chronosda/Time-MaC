import sys
import torch
import torch.nn as nn

# Lazy import transformers components inside initializers to avoid hard deps

# Import custom modules lazily to avoid heavy deps unless needed
sys.path.append("../")
from src.TimeVLM.vlm_custom import CustomVLM

# Optional MAE variants (import lazily in _init_mae to avoid heavy deps)

class VLMManager:
    """
    Manager class to handle different VLM types (CLIP, BLIP2, ViLT).
    """
    def __init__(self, config):
        self.config = config
        self.vlm_type = config.vlm_type.lower()
        self.device = self._acquire_device()
        self.offline = getattr(self.config, 'offline', False)
        self.use_dummy = False
        self._init_vlm()
        if not hasattr(self, 'vision_hidden_size'):
            self.vision_hidden_size = self.hidden_size
        if not hasattr(self, 'text_hidden_size'):
            self.text_hidden_size = self.hidden_size
        
    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_vlm(self):
        if self.vlm_type == "clip":
            self._init_clip()
        elif self.vlm_type == "blip2":
            self._init_blip2()
        elif self.vlm_type == "vilt":
            self._init_vilt()
        elif self.vlm_type == "custom":
            self._init_custom()
        elif self.vlm_type == "mae":
            self._init_mae()
        else:
            raise ValueError(f"Unsupported vlm_type: {self.vlm_type}. Choose from ['clip', 'blip2', 'vilt', 'custom', 'mae'].")
        self.model.to(self.device)
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("VLM Learnable model parameters: {:,}".format(learnable_params))

    def _init_clip(self):
        CLIP_ARCH = 'openai/clip-vit-base-patch32'
        try:
            from transformers import CLIPProcessor, CLIPModel  # lazy import
        except Exception as e:
            if self.offline:
                print(f"Transformers unavailable for CLIP ({e}); using dummy features.")
                self.processor, self.model = None, None
                self.use_dummy = True
                self.hidden_size = 512
                self.fusion_dim = self.hidden_size
                self.max_input_text_length = 77
                self.fused_feature_len = 9
                return
            raise
        try:
            print("Trying to load from local cache...")
            self.processor = CLIPProcessor.from_pretrained(CLIP_ARCH, local_files_only=True)
            self.model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True, local_files_only=True)
            print("Successfully loaded from local cache!")
        except Exception as e:
            print(f"Local cache not found: {e}")
            if self.offline:
                print("Offline mode enabled; using dummy CLIP features.")
                self.processor, self.model = None, None
                self.use_dummy = True
            else:
                print("Loading from remote...")
                self.processor = CLIPProcessor.from_pretrained(CLIP_ARCH)
                self.model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True)
                print("Successfully loaded from remote!")
        
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 512
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77
        self.fused_feature_len = 9
        self.multimodal_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1), 
            nn.Sigmoid()
        ).to(self.device)

    def _init_blip2(self):
        BLIP_ARCH = 'Salesforce/blip2-opt-2.7b'
        try:
            from transformers import Blip2Processor, Blip2Model  # lazy import
        except Exception as e:
            if self.offline:
                print(f"Transformers unavailable for BLIP2 ({e}); using dummy features.")
                self.processor, self.model = None, None
                self.use_dummy = True
                self.hidden_size = 2560
                self.fusion_dim = self.hidden_size
                self.max_input_text_length = 1024
                self.fused_feature_len = 298
                return
            raise
        try:
            print("Trying to load BLIP2 from local cache...")
            self.processor = Blip2Processor.from_pretrained(BLIP_ARCH, local_files_only=True)
            self.model = Blip2Model.from_pretrained(BLIP_ARCH, output_hidden_states=True, local_files_only=True)
            print("Successfully loaded BLIP2 from local cache!")
        except Exception as e:
            print(f"BLIP2 local cache not found: {e}")
            if self.offline:
                print("Offline mode enabled; using dummy BLIP2 features.")
                self.processor, self.model = None, None
                self.use_dummy = True
            else:
                print("Loading BLIP2 from remote...")
                self.processor = Blip2Processor.from_pretrained(BLIP_ARCH)
                self.model = Blip2Model.from_pretrained(BLIP_ARCH, output_hidden_states=True)
                print("Successfully loaded BLIP2 from remote!")
        
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 2560
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 1024
        self.fused_feature_len = 298

    def _init_vilt(self):
        VILT_ARCH = "dandelin/vilt-b32-finetuned-coco"
        try:
            from transformers import ViltProcessor, ViltModel  # lazy import
        except Exception as e:
            if self.offline:
                print(f"Transformers unavailable for ViLT ({e}); using dummy features.")
                self.processor, self.model = None, None
                self.use_dummy = True
                self.hidden_size = 768
                self.fusion_dim = self.hidden_size
                self.max_input_text_length = 40
                self.fused_feature_len = 156
                return
            raise
        try:
            print("Trying to load ViLT from local cache...")
            self.processor = ViltProcessor.from_pretrained(VILT_ARCH, local_files_only=True)
            self.model = ViltModel.from_pretrained(VILT_ARCH, output_hidden_states=True, local_files_only=True)
            print("Successfully loaded ViLT from local cache!")
        except Exception as e:
            print(f"ViLT local cache not found: {e}")
            if self.offline:
                print("Offline mode enabled; using dummy ViLT features.")
                self.processor, self.model = None, None
                self.use_dummy = True
            else:
                print("Loading ViLT from remote...")
                self.processor = ViltProcessor.from_pretrained(VILT_ARCH)
                self.model = ViltModel.from_pretrained(VILT_ARCH, output_hidden_states=True)
                print("Successfully loaded ViLT from remote!")
        
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 768
        if self.config.w_out_query:
            self.fusion_dim = self.hidden_size
        else:
            self.fusion_dim = self.hidden_size
        self.max_input_text_length = 40
        self.fused_feature_len = 156
        
    def _init_custom(self):
        """
        Initialize the custom VLM.
        """
        self.model = CustomVLM(self.config)
        self.hidden_size = self.model.hidden_size
        self.max_input_text_length = 512  # Adjust based on text encoder

    def _init_mae(self):
        """
        Initialize MAE-based encoders, including reconstruction-oriented variants.
        """
        # Flags to select MAE variants
        use_reconstruction = getattr(self.config, 'use_reconstruction_mae', False)
        use_dual_path = getattr(self.config, 'use_dual_path_reconstruction', False)
        use_optimized = getattr(self.config, 'use_optimized_mae', False)

        try:
            if use_dual_path:
                print("Using dual-path reconstruction MAE encoder...")
                from src.TimeVLM.dual_path_reconstruction import DualPathReconstructionVLM
                self.model = DualPathReconstructionVLM(self.config)
            elif use_reconstruction:
                print("Using reconstruction-oriented MAE encoder...")
                from src.TimeVLM.mae_reconstruction_vlm import MAEReconstructionVLM
                self.model = MAEReconstructionVLM(self.config)
            elif use_optimized:
                print("Using optimized MAE encoder for time series data...")
                from src.TimeVLM.mae_encoder_optimized import MAEEncoderOptimized
                self.model = MAEEncoderOptimized(self.config)
            else:
                print("Using standard MAE encoder...")
                from src.TimeVLM.mae_encoder_plugin import MAEEncoderPlugin
                self.model = MAEEncoderPlugin(self.config)
        except Exception as e:
            raise RuntimeError(
                "MAE encoder initialization failed. The main experiment does not "
                "permit silent dummy features."
            ) from e
        self.hidden_size = self.model.hidden_size
        self.vision_hidden_size = getattr(
            self.model,
            'vision_hidden_size',
            self.hidden_size,
        )
        self.text_hidden_size = getattr(
            self.model,
            'text_hidden_size',
            self.hidden_size,
        )
        self.fusion_dim = self.model.fusion_dim
        self.max_input_text_length = self.model.max_input_text_length
        self.fused_feature_len = self.model.fused_feature_len

        # MAE doesn't have multimodal fusion gate like CLIP/BLIP2
        # But we can add a simple one for compatibility
        self.multimodal_fusion_gate = nn.Sequential(
            nn.Linear(
                self.vision_hidden_size + self.text_hidden_size,
                self.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ).to(self.device)

    def _set_requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad = value
        for child in model.children():
            self._set_requires_grad(child, value)

    def process_inputs(self, B, images, prompts):
        try:
            if getattr(self, 'use_dummy', False):
                device = self.device
                return (
                    torch.zeros(B, self.vision_hidden_size, device=device),
                    torch.zeros(B, self.text_hidden_size, device=device)
                )
            if self.vlm_type == "clip":
                return self._process_clip_inputs(B, images, prompts)
            elif self.vlm_type == "blip2":
                return self._process_blip2_inputs(B, images, prompts)
            elif self.vlm_type == "vilt":
                return self._process_vilt_inputs(B, images, prompts)
            elif self.vlm_type == "custom":
                return self._process_custom_inputs(B, images, prompts)
            elif self.vlm_type == "mae":
                return self._process_mae_inputs(B, images, prompts)
        except Exception as e:
            print(f"Error processing inputs: {e}")
            print(f"Images shape: {images.shape}")
            print(f"Prompts: {prompts}")
            raise e

    def encode_text(self, prompts):
        """Optional text encoding; returns zeros in offline/dummy mode."""
        if getattr(self, 'use_dummy', False):
            return torch.zeros(len(prompts), self.text_hidden_size, device=self.device)
        # Minimal implementation to avoid remote calls; prefer process_inputs where possible
        return torch.zeros(len(prompts), self.text_hidden_size, device=self.device)

    def _process_clip_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        text_features = outputs.text_embeds  # Shape: [B, hidden_size]
        image_features = outputs.image_embeds  # Shape: [B, hidden_size]
        return image_features, text_features  # Both shape: [B, hidden_size]

    def _process_blip2_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True).language_model_outputs.hidden_states[-1]   # [B, seq_len, hidden_size]
        
        # Get text token count and extract features
        text_token_count = encoding["input_ids"].shape[1]
        text_features = outputs[:, :text_token_count, :]  # [B, text_token_count, hidden_size]
        image_features = outputs[:, text_token_count:, :]  # [B, seq_len - text_token_count, hidden_size]
        
        # Average pooling for both features
        text_features = text_features.mean(dim=1)  # [B, hidden_size]
        image_features = image_features.mean(dim=1)  # [B, hidden_size]
        
        return image_features, text_features  # Both shape: [B, hidden_size]
    
    def _process_vilt_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state  # Shape: [B, seq_len, hidden_size]
        
        # Get text token count and extract features
        text_token_count = encoding["input_ids"].shape[1]
        text_features = last_hidden_state[:, :text_token_count, :]  # [B, text_token_count, hidden_size]
        image_features = last_hidden_state[:, text_token_count:, :]  # [B, seq_len - text_token_count, hidden_size]
        
        # Average pooling for both features
        text_features = text_features.mean(dim=1)  # [B, hidden_size]
        image_features = image_features.mean(dim=1)  # [B, hidden_size]
        
        return image_features, text_features  # Both shape: [B, hidden_size]
    
    def _process_custom_inputs(self, B, images, prompts):
        vision_embeddings = self.model.get_vision_embeddings(images)    # Shape: [B, hidden_size]
        text_embeddings = self.model.get_text_embeddings(prompts)        # Shape: [B, hidden_size]
        return vision_embeddings, text_embeddings  # Both shape: [B, hidden_size]

    def _process_mae_inputs(self, B, images, prompts):
        """
        Process inputs for MAE encoder plugin.
        """
        # Use the MAE plugin's forward method
        vision_embeddings, text_embeddings = self.model(images, prompts)
        return vision_embeddings, text_embeddings  # Both shape: [B, hidden_size]
