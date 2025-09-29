# MAE Encoder Plugin for TimeVLM

This document describes the integration of a pre-trained Masked Autoencoder (MAE) encoder as a plugin in the ICML25-TimeVLM image module, serving as a replacement for traditional VLM encoders.

## Overview

The MAE encoder plugin provides:
- **Pre-trained Vision Encoding**: Utilizes Facebook's MAE models for powerful vision feature extraction
- **Plugin Architecture**: Drop-in replacement for existing VLM encoders (CLIP, BLIP2, ViLT)
- **Multiple Architectures**: Support for base, large, and huge MAE variants
- **Flexible Fine-tuning**: Configurable fine-tuning modes (full, layer norm, bias, etc.)
- **Time Series Integration**: Seamless integration with TimeVLM's time series to image conversion pipeline

## Architecture

### MAE Encoder Plugin (`mae_encoder_plugin.py`)
- **MAEEncoderPlugin**: Main plugin class for vision encoding
- **MAEEncoderWithDecoder**: Extended version with reconstruction capabilities
- **Preprocessing**: Image preprocessing utilities for MAE compatibility
- **Feature Extraction**: Vision and text feature extraction interfaces

### VLM Manager Integration (`vlm_manager.py`)
- **Modified VLMManager**: Added support for "mae" encoder type
- **Input Processing**: Dedicated MAE input processing method
- **Configuration Support**: MAE-specific configuration parameters

## Key Features

### 1. Multiple MAE Architectures
- **MAE-Base**: 86M parameters, 768 hidden dimensions
- **MAE-Large**: 307M parameters, 1024 hidden dimensions
- **MAE-Huge**: 632M parameters, 1280 hidden dimensions

### 2. Fine-tuning Options
- `full`: Fine-tune all parameters
- `ln`: Fine-tune only layer normalization parameters
- `bias`: Fine-tune only bias parameters
- `none`: No fine-tuning (feature extraction only)
- `mlp`: Fine-tune only MLP layers
- `attn`: Fine-tune only attention layers

### 3. Integration Capabilities
- **Time Series Images**: Compatible with TimeVLM's time series to image conversion
- **Multimodal Fusion**: Supports text modality alongside vision
- **Reconstruction**: Optional decoder for image reconstruction tasks
- **Batch Processing**: Efficient batch processing for multiple images

## Installation

1. **Dependencies**: Ensure you have the required packages:
```bash
pip install torch torchvision transformers timm pillow numpy
```

2. **MAE Checkpoints**: Download pre-trained MAE checkpoints:
```bash
# Create checkpoint directory
mkdir -p ./ckpt/

# Download MAE checkpoints (example for base model)
wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth -O ./ckpt/mae_visualize_vit_base.pth
```

## Usage

### Basic Usage

```python
from src.TimeVLM.vlm_manager import VLMManager
from configs.mae_encoder_config import get_mae_config

# Create configuration
config = get_mae_config("mae_base")

# Initialize VLM manager with MAE encoder
vlm_manager = VLMManager(config)

# Process images
images = [...]  # List of PIL images
prompts = [...]  # List of text prompts

vision_features, text_features = vlm_manager.process_inputs(
    B=len(images),
    images=images,
    prompts=prompts
)
```

### Advanced Configuration

```python
from configs.mae_encoder_config import MAEEncoderConfig

# Custom configuration
config = MAEEncoderConfig(
    vlm_type="mae",
    mae_arch="mae_large",           # Use large MAE
    mae_finetune_type="ln",         # Fine-tune layer norm
    mae_ckpt_dir="./custom_ckpt/",  # Custom checkpoint directory
    finetune_vlm=True,              # Enable fine-tuning
    use_gpu=True,
    gpu=0
)

vlm_manager = VLMManager(config)
```

### Time Series Integration

```python
from layers.TimeSeries_To_Image import time_series_to_simple_image

# Convert time series to images
time_series_data = torch.randn(batch_size, seq_len, n_vars)
images = []
for i in range(batch_size):
    img = time_series_to_simple_image(time_series_data[i].numpy())
    images.append(img)

# Process with MAE encoder
vision_features, text_features = vlm_manager.process_inputs(
    B=len(images),
    images=images,
    prompts=["Time series pattern"] * len(images)
)
```

### Reconstruction Mode

```python
from src.TimeVLM.mae_encoder_plugin import MAEEncoderWithDecoder

# Initialize MAE encoder with decoder
config = get_mae_config("mae_reconstruction")
mae_encoder = MAEEncoderWithDecoder(config)

# Preprocess images
preprocessed_images = mae_encoder.preprocess_images(images)

# Extract features and reconstruct
features, reconstructed_images, mask = mae_encoder.encode_with_reconstruction(
    preprocessed_images, mask_ratio=0.75
)
```

## Configuration Options

### MAE-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mae_arch` | str | "mae_base" | MAE architecture (base/large/huge) |
| `mae_finetune_type` | str | "ln" | Fine-tuning mode (full/ln/bias/none/mlp/attn) |
| `mae_ckpt_dir` | str | "./ckpt/" | Checkpoint directory |
| `mae_load_ckpt` | bool | True | Load pre-trained weights |
| `mae_decoder_enabled` | bool | False | Enable decoder for reconstruction |

### Predefined Configurations

```python
# Base configuration (recommended for most use cases)
config = get_mae_config("mae_base")

# Large configuration (higher accuracy, more compute)
config = get_mae_config("mae_large")

# Huge configuration (highest accuracy, most compute)
config = get_mae_config("mae_huge")

# Fine-tuning configuration
config = get_mae_config("mae_finetune")

# Reconstruction configuration
config = get_mae_config("mae_reconstruction")
```

## Examples

### Running Examples

```bash
# Basic usage example
python examples/mae_encoder_usage.py

# The example includes:
# - Basic MAE encoder usage
# - Time series integration
# - Architecture comparison
# - Fine-tuning demonstration
# - Reconstruction capabilities
# - TimeVLM integration
```

### Example Output

```
=== Basic MAE Encoder Example ===
Initializing MAE encoder: mae_base
Successfully loaded MAE checkpoint: mae_visualize_vit_base.pth
Vision features shape: torch.Size([2, 768])
Text features shape: torch.Size([2, 768])
✓ MAE encoder basic example completed successfully

=== Time Series MAE Encoder Example ===
Vision features shape: torch.Size([4, 768])
Text features shape: torch.Size([4, 768])
Feature dimension: 768
✓ Time series encoder example completed successfully
```

## Performance Considerations

### Model Selection
- **MAE-Base**: Best balance of performance and efficiency
- **MAE-Large**: Higher accuracy, requires more memory
- **MAE-Huge**: Maximum accuracy, requires significant resources

### Fine-tuning Strategy
- **Feature Extraction**: Use `mae_finetune_type="none"` for pure feature extraction
- **Light Fine-tuning**: Use `mae_finetune_type="ln"` for minimal adaptation
- **Full Fine-tuning**: Use `mae_finetune_type="full"` for maximum adaptation

### Memory Usage
- MAE-Base: ~350MB GPU memory
- MAE-Large: ~1.2GB GPU memory
- MAE-Huge: ~2.5GB GPU memory

## Integration with Existing Code

### Replacing VLM Encoders

```python
# Before (using CLIP)
config.vlm_type = "clip"
vlm_manager = VLMManager(config)

# After (using MAE)
config.vlm_type = "mae"
config.mae_arch = "mae_base"
vlm_manager = VLMManager(config)

# The rest of your code remains the same!
vision_features, text_features = vlm_manager.process_inputs(B, images, prompts)
```

### Backward Compatibility
The MAE encoder plugin maintains full backward compatibility with existing TimeVLM code:
- Same input/output interfaces
- Same configuration structure
- Same processing pipeline

## Troubleshooting

### Common Issues

1. **Checkpoint Not Found**
```bash
# Solution: Download the required checkpoint
wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth -O ./ckpt/mae_visualize_vit_base.pth
```

2. **GPU Memory Issues**
```python
# Solution: Use smaller architecture or CPU
config.mae_arch = "mae_base"
config.use_gpu = False
```

3. **Import Errors**
```python
# Solution: Ensure all dependencies are installed
pip install torch torchvision transformers timm pillow numpy
```

### Debug Mode

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model configuration
vlm_manager = VLMManager(config)
print(f"Model config: {vlm_manager.model.get_config()}")
```

## Future Enhancements

### Planned Features
- **Custom MAE Checkpoints**: Support for custom pre-trained MAE models
- **Multi-scale Processing**: Extract features at multiple scales
- **Attention Visualization**: Visualize MAE attention maps
- **Performance Optimization**: Further optimize for time series data

### Research Directions
- **Time Series Specific Pre-training**: Pre-train MAE on time series data
- **Adaptive Masking**: Dynamic masking strategies for time series
- **Multimodal Fusion**: Enhanced vision-text fusion for time series analysis

## References

- **MAE Paper**: He, K., et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
- **TimeVLM**: Original TimeVLM implementation and documentation
- **Vision Transformers**: Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." ICLR 2021.

## License

This implementation follows the same license as the original TimeVLM project. The MAE models are subject to their respective licenses.