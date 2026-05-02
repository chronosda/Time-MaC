#!/usr/bin/env python3
import torch
from PIL import Image

from configs.config import TimeMEConfig
from src.TimeVLM.vlm_manager import VLMManager


def make_dummy_images(batch=2, size=224):
    # Create simple gradient images
    imgs = torch.linspace(0, 1, steps=size).repeat(size, 1)
    imgs = imgs.unsqueeze(0).repeat(3, 1, 1)  # 3xHxW
    batch_imgs = imgs.unsqueeze(0).repeat(batch, 1, 1, 1)
    return batch_imgs


def run_variant(name, config_overrides=None):
    cfg = TimeMEConfig()
    cfg.vlm_type = 'mae'
    cfg.use_gpu = False
    cfg.offline = True
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)
    print(f"\n=== Testing {name} ===")
    vlm = VLMManager(cfg)
    images = make_dummy_images()
    prompts = ["dummy"] * images.shape[0]
    vision, text = vlm.process_inputs(images.shape[0], images, prompts)
    print("vision:", tuple(vision.shape), "text:", tuple(text.shape))


if __name__ == "__main__":
    # Standard MAE
    run_variant("standard MAE", {})
    # Reconstruction-oriented MAE
    run_variant("reconstruction-oriented MAE", {"use_reconstruction_mae": True})
    # Optimized MAE
    run_variant("optimized MAE", {"use_optimized_mae": True})
    # Dual-path reconstruction MAE
    run_variant("dual-path reconstruction", {"use_dual_path_reconstruction": True})

