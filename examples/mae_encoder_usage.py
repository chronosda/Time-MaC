"""
Example usage of MAE encoder plugin in TimeVLM.
This example demonstrates how to use MAE encoder as a replacement for VLM encoders.
"""

import torch
import sys
from PIL import Image
import numpy as np

# Add the project root to the path
sys.path.append("../")

from src.TimeVLM.vlm_manager import VLMManager
from configs.mae_encoder_config import MAEEncoderConfig, get_mae_config


def create_dummy_images(batch_size=4, image_size=224):
    """
    Create dummy images for testing.

    Args:
        batch_size: Number of images to create
        image_size: Size of the images (square)

    Returns:
        List[PIL.Image]: List of dummy PIL images
    """
    images = []
    for i in range(batch_size):
        # Create a simple pattern
        img_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    return images


def create_time_series_images():
    """
    Create time series images using the TimeVLM conversion methods.
    This demonstrates integration with the existing TimeVLM pipeline.
    """
    try:
        from layers.TimeSeries_To_Image import time_series_to_simple_image

        # Create dummy time series data
        seq_len = 96
        n_vars = 1
        batch_size = 4

        # Generate synthetic time series data
        time_series_data = torch.randn(batch_size, seq_len, n_vars)

        # Convert to images
        images = []
        for i in range(batch_size):
            ts_data = time_series_data[i].numpy()
            img = time_series_to_simple_image(ts_data)
            images.append(img)

        return images

    except ImportError:
        print("Warning: Time series conversion not available, using dummy images")
        return create_dummy_images()


def example_mae_encoder_basic():
    """
    Basic example of using MAE encoder plugin.
    """
    print("=== Basic MAE Encoder Example ===")

    # Create configuration
    config = get_mae_config("mae_base")

    # Initialize VLM manager with MAE encoder
    vlm_manager = VLMManager(config)

    # Create dummy images
    images = create_dummy_images(batch_size=2)

    # Create dummy prompts
    prompts = ["This is a test image", "Another test image"]

    # Process inputs
    try:
        vision_features, text_features = vlm_manager.process_inputs(
            B=len(images),
            images=images,
            prompts=prompts
        )

        print(f"Vision features shape: {vision_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Hidden size: {vlm_manager.hidden_size}")

        print("✓ MAE encoder basic example completed successfully")

    except Exception as e:
        print(f"✗ Error in basic example: {e}")


def example_mae_encoder_time_series():
    """
    Example of using MAE encoder with time series images.
    """
    print("\n=== Time Series MAE Encoder Example ===")

    # Create configuration
    config = get_mae_config("mae_base")

    # Set time series specific parameters
    config.seq_len = 96
    config.pred_len = 96
    config.task_name = "long_term_forecast"

    # Initialize VLM manager with MAE encoder
    vlm_manager = VLMManager(config)

    # Create time series images
    images = create_time_series_images()

    # Create time series related prompts
    prompts = [
        "Time series forecasting pattern",
        "Historical data visualization",
        "Seasonal trend analysis",
        "Predictive modeling data"
    ]

    # Process inputs
    try:
        vision_features, text_features = vlm_manager.process_inputs(
            B=len(images),
            images=images,
            prompts=prompts
        )

        print(f"Vision features shape: {vision_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Feature dimension: {vlm_manager.hidden_size}")

        print("✓ Time series MAE encoder example completed successfully")

    except Exception as e:
        print(f"✗ Error in time series example: {e}")


def example_mae_encoder_architectures():
    """
    Example of using different MAE architectures.
    """
    print("\n=== MAE Architecture Comparison ===")

    architectures = ["mae_base", "mae_large", "mae_huge"]

    for arch in architectures:
        try:
            print(f"\n--- Testing {arch} ---")

            # Create configuration
            config = get_mae_config(arch)

            # Initialize VLM manager
            vlm_manager = VLMManager(config)

            # Create dummy images
            images = create_dummy_images(batch_size=1)
            prompts = ["Test image"]

            # Process inputs
            vision_features, text_features = vlm_manager.process_inputs(
                B=len(images),
                images=images,
                prompts=prompts
            )

            print(f"✓ {arch}: Hidden size = {vlm_manager.hidden_size}")

        except Exception as e:
            print(f"✗ Error with {arch}: {e}")


def example_mae_encoder_finetuning():
    """
    Example of using MAE encoder with fine-tuning.
    """
    print("\n=== MAE Fine-tuning Example ===")

    # Create configuration with fine-tuning enabled
    config = get_mae_config("mae_finetune")

    # Initialize VLM manager
    vlm_manager = VLMManager(config)

    # Create dummy images
    images = create_dummy_images(batch_size=2)
    prompts = ["Training image 1", "Training image 2"]

    # Process inputs
    try:
        vision_features, text_features = vlm_manager.process_inputs(
            B=len(images),
            images=images,
            prompts=prompts
        )

        print(f"Vision features shape: {vision_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Fine-tuning enabled: {config.finetune_vlm}")

        # Check if parameters are trainable
        trainable_params = sum(p.numel() for p in vlm_manager.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in vlm_manager.model.parameters())

        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params:.2%}")

        print("✓ MAE fine-tuning example completed successfully")

    except Exception as e:
        print(f"✗ Error in fine-tuning example: {e}")


def example_mae_encoder_reconstruction():
    """
    Example of using MAE encoder with reconstruction capabilities.
    """
    print("\n=== MAE Reconstruction Example ===")

    # Create configuration with decoder enabled
    config = get_mae_config("mae_reconstruction")

    # Import the MAE encoder with decoder
    from src.TimeVLM.mae_encoder_plugin import MAEEncoderWithDecoder

    # Initialize the encoder with decoder
    mae_encoder = MAEEncoderWithDecoder(config)

    # Create dummy images
    images = create_dummy_images(batch_size=1)

    # Preprocess images
    preprocessed_images = mae_encoder.preprocess_images(images)

    try:
        # Extract features and reconstruct images
        features, reconstructed_images, mask = mae_encoder.encode_with_reconstruction(
            preprocessed_images, mask_ratio=0.75
        )

        print(f"Original images shape: {preprocessed_images.shape}")
        print(f"Reconstructed images shape: {reconstructed_images.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask ratio: {mask.mean().item():.2%}")

        print("✓ MAE reconstruction example completed successfully")

    except Exception as e:
        print(f"✗ Error in reconstruction example: {e}")


def example_integration_with_timevlm():
    """
    Example of integrating MAE encoder with the main TimeVLM model.
    """
    print("\n=== TimeVLM Integration Example ===")

    try:
        from src.TimeVLM.model import Model

        # Create configuration
        config = get_mae_config("mae_base")
        config.seq_len = 96
        config.pred_len = 96
        config.task_name = "long_term_forecast"

        # Initialize TimeVLM model with MAE encoder
        model = Model(config)

        print("✓ TimeVLM model with MAE encoder initialized successfully")
        print(f"Model configuration: {model.config}")

    except Exception as e:
        print(f"✗ Error in TimeVLM integration: {e}")


def main():
    """
    Run all examples.
    """
    print("MAE Encoder Plugin Examples")
    print("=" * 50)

    # Run examples
    example_mae_encoder_basic()
    example_mae_encoder_time_series()
    example_mae_encoder_architectures()
    example_mae_encoder_finetuning()
    example_mae_encoder_reconstruction()
    example_integration_with_timevlm()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()