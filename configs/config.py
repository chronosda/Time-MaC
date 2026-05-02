import argparse
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TimeMEConfig:
    """Configuration for Time-me model"""
    # Basic model parameters
    d_model: int = 512
    pred_len: int = 96
    seq_len: int = 512
    enc_in: int = 7
    c_out: int = 7

    # Enhanced fusion parameters
    use_enhanced_fusion: bool = True
    mamba_layers: int = 2
    use_mae_vision: bool = False
    # Use TimeBase as temporal backbone (complexity reducer)
    use_timebase_backbone: bool = False

    # Vision parameters
    image_size: int = 224
    three_channel_image: bool = True
    patch_len: int = 16
    stride: int = 8
    padding: int = 0

    # Memory parameters
    patch_memory_size: int = 100
    top_k: int = 5
    num_attention_heads: int = 8

    # Training parameters
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100

    # Device parameters
    use_gpu: bool = True
    gpu: int = 0
    # Offline mode to avoid remote model downloads
    offline: bool = False

    # Task parameters
    task_name: str = 'long_term_forecast'
    periodicity: int = 24

    # Mamba parameters
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    e_layers: int = 2

    # Embedding parameters
    embed: str = 'timeF'
    freq: str = 'h'
    d_ff: int = 2048
    # DataLoader workers
    num_workers: int = 0

    # VLM parameters
    vlm_type: str = 'clip'  # options: clip, blip2, vilt, custom, mae
    finetune_vlm: bool = False
    mae_arch: str = 'mae_base'
    mae_finetune_type: str = 'ln'
    mae_ckpt_dir: str = './ckpt/'
    mae_load_ckpt: bool = False

    # Reconstruction-oriented MAE flags
    use_reconstruction_mae: bool = False
    use_dual_path_reconstruction: bool = False
    use_optimized_mae: bool = False

    # Reconstruction MAE hyperparameters
    reconstruction_ratio: float = 0.3
    use_reconstruction_features: bool = True
    multimodal_fusion_type: str = 'reconstruction_aware'  # or 'simple'
    use_vlm_path: bool = True
    use_reconstruction_path: bool = True
    path_fusion_strategy: str = 'adaptive'  # adaptive|fixed|learned
    reconstruction_strength: float = 0.7

    # Optimized MAE options
    use_adaptive_norm: bool = True
    use_global_features: bool = True
    feature_fusion: bool = True

    # Data shaping
    restrict_vars: int = -1  # if >0, use only first K variables (columns)
    data_split: str = 'ratio'  # ratio|ett_standard|custom_standard
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    calib_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 0
    train_subset_ratio: float = 1.0
    train_subset_seed: int = 0

    # Conformal calibration (post-hoc uncertainty)
    conformal_enable: bool = False
    conformal_method: str = 'crc'  # crc|hpd|rcps
    conformal_alpha: float = 0.10
    conformal_scale: str = 'mad'   # mad|std|global_mad
    conformal_hpd_level: float = 0.95
    conformal_num_dir: int = 1000
    conformal_delta: float = 0.05
    conformal_max_threshold: float = -1.0  # <=0 means auto
    conformal_mode: str = 'online'  # online|static
    conformal_update_lr: float = 0.005
    conformal_buffer_size: int = 4096
    conformal_alpha_min: float = 0.005
    conformal_alpha_max: float = 0.30
    conformal_recompute_method: str = 'bq'  # bq|quantile
    conformal_update_block_size: int = 8


def create_config_from_args(args):
    """Create TimeMEConfig from argparse arguments"""
    config = TimeMEConfig()

    # Update config with args
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Set device configuration
    config.device = f'cuda:{config.gpu}' if config.use_gpu and torch.cuda.is_available() else 'cpu'

    return config


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Time-MaC: time-series forecasting with Coupled-Mamba fusion')

    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')

    # Enhanced fusion parameters
    parser.add_argument('--use_enhanced_fusion', action='store_true', help='use enhanced coupled-mamba fusion')
    parser.add_argument('--mamba_layers', type=int, default=2, help='number of mamba layers')
    parser.add_argument('--use_mae_vision', action='store_true', help='use MAE for vision encoding')
    parser.add_argument('--use_timebase_backbone', action='store_true',
                        help='use TimeBase as temporal backbone to reduce temporal complexity')

    # Vision parameters
    parser.add_argument('--image_size', type=int, default=224, help='image size for vision encoding')
    parser.add_argument('--three_channel_image', action='store_true', help='use three channel images')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride for patch embedding')
    parser.add_argument('--padding', type=int, default=0, help='padding for patch embedding')

    # Memory parameters
    parser.add_argument('--patch_memory_size', type=int, default=100, help='patch memory bank size')
    parser.add_argument('--top_k', type=int, default=5, help='top-k for patch retrieval')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')

    # Training parameters
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    # Device parameters
    parser.add_argument('--use_gpu', action='store_true', help='use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')

    # Task parameters
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
    parser.add_argument('--periodicity', type=int, default=24, help='data periodicity')

    # Mamba parameters
    parser.add_argument('--d_state', type=int, default=64, help='mamba state dimension')
    parser.add_argument('--d_conv', type=int, default=4, help='mamba convolution kernel size')
    parser.add_argument('--expand', type=int, default=2, help='mamba expansion factor')
    parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')

    # Embedding parameters
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--freq', type=str, default='h', help='frequency for time features')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker count')

    # VLM parameters
    parser.add_argument('--vlm_type', type=str, default='clip', help='VLM type: clip|blip2|vilt|custom|mae')
    parser.add_argument('--finetune_vlm', action='store_true', help='finetune VLM parameters')
    parser.add_argument('--offline', action='store_true', help='run in offline mode (no remote downloads)')
    # MAE checkpoint controls
    parser.add_argument('--mae_arch', type=str, default='mae_base', help='MAE architecture: mae_base|mae_large|mae_huge')
    parser.add_argument('--mae_finetune_type', type=str, default='ln', help='MAE finetune type: none|ln|bias|mlp|attn|full|enhanced|adaptive')
    parser.add_argument('--mae_ckpt_dir', type=str, default='./ckpt/', help='directory containing MAE checkpoints')
    parser.add_argument('--mae_load_ckpt', action='store_true', help='load MAE checkpoint weights')

    # Reconstruction-oriented MAE options
    parser.add_argument('--use_reconstruction_mae', action='store_true', help='use reconstruction-oriented MAE encoder')
    parser.add_argument('--use_dual_path_reconstruction', action='store_true', help='use dual-path reconstruction VLM')
    parser.add_argument('--use_optimized_mae', action='store_true', help='use optimized MAE encoder for time series')
    parser.add_argument('--reconstruction_ratio', type=float, default=0.3, help='mask ratio for MAE reconstruction')
    parser.add_argument('--use_reconstruction_features', action='store_true', help='use reconstruction features in fusion')
    parser.add_argument('--multimodal_fusion_type', type=str, default='reconstruction_aware', help='fusion: reconstruction_aware|simple')
    parser.add_argument('--use_vlm_path', action='store_true', help='enable standard VLM path in dual-path encoder')
    parser.add_argument('--use_reconstruction_path', action='store_true', help='enable MAE reconstruction path in dual-path encoder')
    parser.add_argument('--path_fusion_strategy', type=str, default='adaptive', help='path fusion: adaptive|fixed|learned')
    parser.add_argument('--reconstruction_strength', type=float, default=0.7, help='fixed fusion weight for reconstruction path')

    # Optimized MAE toggles
    parser.add_argument('--use_adaptive_norm', action='store_true', help='enable adaptive normalization for MAE features')
    parser.add_argument('--use_global_features', action='store_true', help='use global pooled patch features')
    parser.add_argument('--feature_fusion', action='store_true', help='fuse CLS and global features')

    # Data and output
    parser.add_argument('--data', type=str, required=True, help='dataset name (without .csv)')
    parser.add_argument('--root_path', type=str, required=True, help='root path of data file or dataset directory')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='path to save model')
    # Data shaping
    parser.add_argument('--restrict_vars', type=int, default=-1, help='use only first K variables if >0')
    parser.add_argument('--data_split', type=str, default='ratio',
                        choices=['ratio', 'ett_standard', 'custom_standard'],
                        help='data split protocol: ratio, standard ETT benchmark split, or standard custom benchmark split')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation split ratio')
    parser.add_argument('--calib_ratio', type=float, default=0.1, help='calibration split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test split ratio')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--train_subset_ratio', type=float, default=1.0,
                        help='fraction of train windows to keep after splitting; keeps val/calib/test fixed')
    parser.add_argument('--train_subset_seed', type=int, default=0,
                        help='seed used when subsampling train windows')

    # Conformal calibration flags
    parser.add_argument('--conformal_enable', action='store_true', help='enable conformal calibration')
    parser.add_argument('--conformal_method', type=str, default='crc', help='conformal method: crc|hpd|rcps')
    parser.add_argument('--conformal_alpha', type=float, default=0.10, help='target miscoverage (1-coverage)')
    parser.add_argument('--conformal_scale', type=str, default='mad', help='scale proxy: mad|std|global_mad')
    parser.add_argument('--conformal_hpd_level', type=float, default=0.95, help='HPD level (for hpd)')
    parser.add_argument('--conformal_num_dir', type=int, default=1000, help='Dirichlet samples (for hpd)')
    parser.add_argument('--conformal_delta', type=float, default=0.05, help='delta for Hoeffding UCB (for rcps)')
    parser.add_argument('--conformal_max_threshold', type=float, default=-1.0, help='max threshold upper bound; <=0 auto')
    parser.add_argument('--conformal_mode', type=str, default='online',
                        choices=['online', 'static'],
                        help='conformal mode: online ACI-inspired calibration or static global threshold')
    parser.add_argument('--conformal_update_lr', type=float, default=0.005,
                        help='online conformal update rate for adaptive alpha_t')
    parser.add_argument('--conformal_buffer_size', type=int, default=4096,
                        help='recent score buffer size per horizon for online conformal')
    parser.add_argument('--conformal_alpha_min', type=float, default=0.005,
                        help='minimum horizon-wise alpha_t for online conformal')
    parser.add_argument('--conformal_alpha_max', type=float, default=0.30,
                        help='maximum horizon-wise alpha_t for online conformal')
    parser.add_argument('--conformal_recompute_method', type=str, default='bq',
                        choices=['bq', 'quantile'],
                        help='online threshold recomputation: Bayesian-style solver or empirical quantile')
    parser.add_argument('--conformal_update_block_size', type=int, default=8,
                        help='number of chronological batches per online conformal update block')

    # Evaluation-only robustness controls
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='directory containing saved checkpoints for robustness evaluation')
    parser.add_argument('--output_json', type=str, default='',
                        help='optional JSON path for robustness evaluation output')
    parser.add_argument('--eval_noise_std', type=float, default=0.0,
                        help='Gaussian noise std added to inputs during robustness evaluation')
    parser.add_argument('--eval_mask_ratio', type=float, default=0.0,
                        help='fraction of input steps masked during robustness evaluation')
    parser.add_argument('--eval_mask_mode', type=str, default='random',
                        help='masking mode for robustness evaluation: random|block')
    parser.add_argument('--eval_mask_block_len', type=int, default=16,
                        help='block length when eval_mask_mode=block')
    parser.add_argument('--eval_mask_value', type=float, default=0.0,
                        help='fill value used for masked inputs during robustness evaluation')

    return parser.parse_args()
