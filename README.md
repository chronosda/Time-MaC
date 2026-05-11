# Time-MaC

Time-MaC is a multimodal time-series forecasting project centered on masked autoencoding, coupled Mamba fusion, robustness evaluation, and conformal uncertainty calibration. The repository keeps the forecasting benchmark scaffolding from the original research codebase, but the main development target is the Time-MaC/Time-me model family in this project.

![Time-MaC overview](docs/figures/time_me_overview_vanilla_v2_refined.png)

## Highlights

- Coupled Mamba fusion for temporal, visual, and text-side representations.
- MAE-based reconstruction and optimized encoder variants for time-series image features.
- Robustness experiments for missingness, perturbation, low-resource training, and conformal calibration.
- Public benchmark scripts for ETT, electricity, weather, and traffic forecasting settings.
- Lightweight offline mode for environments where VLM checkpoints cannot be downloaded.

## Repository Layout

```text
Time-MaC/
├── configs/              # Time-MaC and MAE experiment configuration
├── data_provider/        # Benchmark data loaders from the forecasting scaffold
├── docs/                 # Project notes, experiment plans, and figures
├── exp/                  # Forecasting experiment runners
├── layers/               # Attention, embedding, image conversion, and MAE layers
├── models/               # Time-MaC model, baselines, and adapters
├── paper/                # Manuscript drafts, method text, and bibliography
├── scripts/              # Training, benchmark, ablation, and visualization scripts
├── src/                  # Coupled fusion and VLM-compatible components
├── utils/                # Metrics, data loading, perturbation, and conformal utilities
├── run.py                # General benchmark entry point
├── train_traffic.py      # Traffic-focused training entry point
└── test_model_simple.py  # Fast smoke test without external VLM downloads
```

Large datasets, checkpoints, logs, generated prediction arrays, and local caches are intentionally excluded from git.

## Setup

Create an environment with Python 3.9+ for the full Coupled-Mamba path. Install PyTorch first so `mamba-ssm` can compile or select a compatible wheel against the active Torch/CUDA environment:

```bash
conda create -n time-mac python=3.10
conda activate time-mac

# Install the PyTorch build that matches your CUDA driver first.
# Example only; adjust the CUDA index URL for your machine.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install --no-build-isolation -r requirements.txt
```

If you only want to run the lightweight smoke test on CPU, the project has a Transformer fallback inside `src/coupled_mamba_fusion.py` when `mamba-ssm` is unavailable or CUDA is not active. For real Coupled-Mamba training, install `mamba-ssm` with CUDA support.

The key Mamba-related packages are:

- `mamba-ssm[causal-conv1d]`: selective state-space kernel used by Coupled-Mamba and the Mamba baseline.
- `ninja` and `packaging`: build helpers commonly needed when compiling Mamba/CUDA extensions.

General dependency installation:

```bash
pip install -r requirements.txt
```

For CUDA training, prefer the `--no-build-isolation` command above after PyTorch is installed. This avoids building `mamba-ssm` in an isolated environment that cannot see the active Torch installation.

## Quick Checks

Run a lightweight model smoke test:

```bash
python test_model_simple.py
```

Run the full model test only when required VLM/MAE assets are available locally or network access is enabled:

```bash
python test_model.py
```

## Training Examples

Train the Time-MaC/Time-me model through the project-specific script:

```bash
python scripts/train_time_me.py \
  --data ETTm2 \
  --root_path ./dataset/ETT-small \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32
```

Run a public benchmark batch:

```bash
bash scripts/run_public_benchmark_all.sh
```

Run robustness and missingness ablations:

```bash
bash scripts/run_ablation_robustness.sh
bash scripts/run_ablation_strong_missing.sh
```

## Data and Checkpoints

Place benchmark datasets under `dataset/` and model checkpoints under `checkpoints/` or `ckpt/` as needed. These directories are ignored by git because they can be very large.

Recommended local layout:

```text
dataset/
├── ETT-small/
├── electricity/
├── traffic/
└── weather/

checkpoints/
ckpt/
logs/
predictions/
```

## Main Components

- `models/time_me.py`: Time-MaC/Time-me model wrapper.
- `src/coupled_mamba_fusion.py`: coupled Mamba multimodal fusion module.
- Optimized MAE encoder modules under `src/`: reconstruction-oriented visual features.
- `utils/conformal_plugin.py`: static and adaptive conformal calibration.
- `utils/input_perturb.py`: perturbation utilities for robustness testing.
- `scripts/eval_time_me_robustness.py`: robustness evaluation driver.
- `scripts/visualize_intervals.py`: conformal interval visualization.

## Notes

This repository includes baseline forecasting models and experiment runners so Time-MaC can be compared against common time-series architectures. The documentation and default entry points are organized around the Time-MaC work rather than the upstream scaffold.
