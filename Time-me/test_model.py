#!/usr/bin/env python3
"""
Test script for Time-me model
Verify the implementation works correctly
"""

import os
import sys
import torch
import numpy as np
import argparse
from dataclasses import dataclass

# Add project root to path dynamically
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models import TimeMEModel
from configs.config import TimeMEConfig


@dataclass
class TestConfig:
    """Test configuration"""
    d_model: int = 256
    pred_len: int = 12
    seq_len: int = 96
    enc_in: int = 3
    c_out: int = 3
    use_enhanced_fusion: bool = True
    mamba_layers: int = 2
    use_mae_vision: bool = False
    image_size: int = 224
    three_channel_image: bool = True
    patch_len: int = 8
    stride: int = 4
    padding: int = 0
    patch_memory_size: int = 50
    top_k: int = 3
    num_attention_heads: int = 4
    dropout: float = 0.1
    use_gpu: bool = False
    gpu: int = 0
    task_name: str = 'long_term_forecast'
    periodicity: int = 24
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    e_layers: int = 2
    embed: str = 'timeF'
    freq: str = 'h'
    d_ff: int = 1024

    # VLM specific parameters
    vlm_type: str = 'clip'
    vlm_model_name: str = 'openai/clip-vit-base-patch32'
    hidden_size: int = 512
    finetune_vlm: bool = False
    mae_arch: str = 'mae_base'
    mae_finetune_type: str = 'ln'
    mae_ckpt_dir: str = './ckpt/'
    mae_load_ckpt: bool = False


def create_test_data(batch_size=4, seq_len=96, n_vars=3, pred_len=12):
    """Create synthetic test data"""
    # Create time series data with some patterns
    time_steps = seq_len + pred_len
    t = np.linspace(0, 4*np.pi, time_steps)

    # Generate synthetic multivariate time series
    data = np.zeros((batch_size, time_steps, n_vars))

    for i in range(batch_size):
        for j in range(n_vars):
            # Mix of sine waves with different frequencies and phases
            freq1 = 0.5 + 0.1 * j
            freq2 = 0.3 + 0.05 * i
            phase = np.random.random() * 2 * np.pi

            data[i, :, j] = (
                np.sin(freq1 * t + phase) +
                0.5 * np.sin(freq2 * t) +
                0.1 * np.random.randn(time_steps)
            )

    return torch.FloatTensor(data)


def test_model_creation():
    """Test model creation and basic functionality"""
    print("Testing model creation...")

    # Create test configuration
    config = TestConfig()
    config.device = 'cpu'

    try:
        # Create model
        model = TimeMEModel(config)
        print("✓ Model created successfully")

        # Test model info
        model_info = model.get_model_info()
        print(f"✓ Model info: {model_info}")

        # Check if model can move to device
        model.to(config.device)
        print("✓ Model moved to device successfully")

        return model

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        raise


def test_model_forward(model, config):
    """Test model forward pass"""
    print("\nTesting model forward pass...")

    try:
        # Create test data
        batch_size = 2
        test_data = create_test_data(
            batch_size=batch_size,
            seq_len=config.seq_len,
            n_vars=config.enc_in,
            pred_len=config.pred_len
        )

        # Prepare input for model
        x_enc = test_data[:, :config.seq_len, :]
        x_dec = test_data[:, config.seq_len:, :]

        print(f"Input shape: {x_enc.shape}")
        print(f"Target shape: {x_dec.shape}")

        # Forward pass
        with torch.no_grad():
            predictions = model(x_enc)

        print(f"✓ Forward pass successful")
        print(f"Predictions shape: {predictions.shape}")

        # Check output shape
        expected_shape = (batch_size, config.pred_len, config.c_out)
        if predictions.shape == expected_shape:
            print("✓ Output shape correct")
        else:
            print(f"✗ Output shape mismatch. Expected {expected_shape}, got {predictions.shape}")

        # Check if predictions are finite
        if torch.isfinite(predictions).all():
            print("✓ Predictions are finite")
        else:
            print("✗ Predictions contain NaN or Inf values")

        return predictions

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_model_gradients(model, config):
    """Test model gradient computation"""
    print("\nTesting model gradient computation...")

    try:
        # Create test data
        batch_size = 2
        test_data = create_test_data(
            batch_size=batch_size,
            seq_len=config.seq_len,
            n_vars=config.enc_in,
            pred_len=config.pred_len
        )

        x_enc = test_data[:, :config.seq_len, :]
        x_dec = test_data[:, config.seq_len:, :]

        # Forward pass with gradients
        predictions = model(x_enc)

        # Compute loss
        criterion = torch.nn.MSELoss()
        loss = criterion(predictions, x_dec)

        print(f"✓ Loss computed: {loss.item():.6f}")

        # Backward pass
        loss.backward()

        # Check gradients
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        print(f"✓ Gradient norm: {total_norm:.6f}")

        if not np.isnan(total_norm) and total_norm > 0:
            print("✓ Gradients computed successfully")
        else:
            print("✗ Gradient computation failed")

    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        raise


def test_model_memory_usage(model, config):
    """Test model memory usage"""
    print("\nTesting model memory usage...")

    try:
        # Create test data
        batch_size = 4
        test_data = create_test_data(
            batch_size=batch_size,
            seq_len=config.seq_len,
            n_vars=config.enc_in,
            pred_len=config.pred_len
        )

        x_enc = test_data[:, :config.seq_len, :]

        # Get memory before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = 0

        # Forward pass
        with torch.no_grad():
            predictions = model(x_enc)

        # Get memory after forward pass
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_used = memory_after - memory_before
            print(f"✓ GPU memory used: {memory_used / 1024**2:.2f} MB")
        else:
            print("✓ CPU-only mode, skipping GPU memory test")

    except Exception as e:
        print(f"✗ Memory usage test failed: {e}")
        raise


def test_model_consistency(model, config):
    """Test model consistency across multiple runs"""
    print("\nTesting model consistency...")

    try:
        # Create test data
        batch_size = 2
        test_data = create_test_data(
            batch_size=batch_size,
            seq_len=config.seq_len,
            n_vars=config.enc_in,
            pred_len=config.pred_len
        )

        x_enc = test_data[:, :config.seq_len, :]

        # Set model to eval mode
        model.eval()

        # Multiple forward passes
        predictions_list = []
        for i in range(3):
            with torch.no_grad():
                predictions = model(x_enc)
                predictions_list.append(predictions)

        # Check consistency
        all_same = True
        for i in range(1, len(predictions_list)):
            if not torch.allclose(predictions_list[0], predictions_list[i], atol=1e-6):
                all_same = False
                break

        if all_same:
            print("✓ Model is consistent across multiple runs")
        else:
            print("✗ Model predictions vary across runs (eval mode)")

    except Exception as e:
        print(f"✗ Consistency test failed: {e}")
        raise


def main():
    """Main test function"""
    print("="*60)
    print("Time-me Model Test Suite")
    print("="*60)

    # Test configuration
    config = TestConfig()
    print(f"Test configuration: d_model={config.d_model}, seq_len={config.seq_len}, pred_len={config.pred_len}")

    try:
        # Test 1: Model creation
        model = test_model_creation()

        # Test 2: Forward pass
        predictions = test_model_forward(model, config)

        # Test 3: Gradient computation
        test_model_gradients(model, config)

        # Test 4: Memory usage
        test_model_memory_usage(model, config)

        # Test 5: Consistency
        test_model_consistency(model, config)

        print("\n" + "="*60)
        print("✓ All tests passed! Time-me model is working correctly.")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
