"""
Configuration example for MAE encoder plugin in TimeVLM.
This config demonstrates how to use MAE encoder as a replacement for VLM encoders.
"""

class MAEEncoderConfig:
    """
    Configuration class for MAE encoder plugin.
    """

    # Basic configuration
    vlm_type = "mae"  # Use MAE encoder plugin

    # MAE specific settings
    mae_arch = "mae_base"  # Choose from: "mae_base", "mae_large", "mae_huge"
    mae_finetune_type = "ln"  # Choose from: "full", "ln", "bias", "none", "mlp", "attn"
    mae_ckpt_dir = "./ckpt/"
    mae_load_ckpt = True
    mae_decoder_enabled = False  # Set to True if you need reconstruction capabilities

    # General VLM settings
    finetune_vlm = False  # Set to True to fine-tune MAE encoder

    # GPU settings
    use_gpu = True
    gpu = 0

    # TimeVLM integration settings
    task_name = "long_term_forecast"
    seq_len = 96
    pred_len = 96
    norm_const = 0.4
    periodicity = 1
    align_const = 0.4
    interpolation = 'bilinear'

    def __init__(self, **kwargs):
        """
        Initialize configuration with optional overrides.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config key '{key}'")

    @classmethod
    def get_base_config(cls):
        """Get base MAE configuration."""
        return cls(
            vlm_type="mae",
            mae_arch="mae_base",
            mae_finetune_type="ln",
            mae_load_ckpt=True,
            mae_decoder_enabled=False,
            finetune_vlm=False
        )

    @classmethod
    def get_large_config(cls):
        """Get large MAE configuration."""
        return cls(
            vlm_type="mae",
            mae_arch="mae_large",
            mae_finetune_type="ln",
            mae_load_ckpt=True,
            mae_decoder_enabled=False,
            finetune_vlm=False
        )

    @classmethod
    def get_huge_config(cls):
        """Get huge MAE configuration."""
        return cls(
            vlm_type="mae",
            mae_arch="mae_huge",
            mae_finetune_type="ln",
            mae_load_ckpt=True,
            mae_decoder_enabled=False,
            finetune_vlm=False
        )

    @classmethod
    def get_finetune_config(cls):
        """Get MAE configuration with fine-tuning enabled."""
        return cls(
            vlm_type="mae",
            mae_arch="mae_base",
            mae_finetune_type="ln",
            mae_load_ckpt=True,
            mae_decoder_enabled=False,
            finetune_vlm=True
        )

    @classmethod
    def get_reconstruction_config(cls):
        """Get MAE configuration with decoder enabled for reconstruction."""
        return cls(
            vlm_type="mae",
            mae_arch="mae_base",
            mae_finetune_type="ln",
            mae_load_ckpt=True,
            mae_decoder_enabled=True,
            finetune_vlm=False
        )


# Example usage configurations
EXAMPLE_CONFIGS = {
    "mae_base": MAEEncoderConfig.get_base_config(),
    "mae_large": MAEEncoderConfig.get_large_config(),
    "mae_huge": MAEEncoderConfig.get_huge_config(),
    "mae_finetune": MAEEncoderConfig.get_finetune_config(),
    "mae_reconstruction": MAEEncoderConfig.get_reconstruction_config(),
}


def get_mae_config(config_name="mae_base"):
    """
    Get a predefined MAE configuration.

    Args:
        config_name: Name of the configuration ("mae_base", "mae_large", "mae_huge",
                    "mae_finetune", "mae_reconstruction")

    Returns:
        MAEEncoderConfig: The requested configuration
    """
    if config_name not in EXAMPLE_CONFIGS:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(EXAMPLE_CONFIGS.keys())}")

    return EXAMPLE_CONFIGS[config_name]