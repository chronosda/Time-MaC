# Time-me: Enhanced Time-VLM with Coupled-Mamba Fusion

This project improves Time-VLM by properly integrating the coupled-mamba multimodal fusion architecture.

## Architecture

The enhanced model combines:
- Original Time-VLM for temporal and vision-language processing
- Coupled-Mamba fusion for improved multimodal integration
- Fixed implementation addressing interface mismatches and architectural issues

## Key Improvements

1. **Correct MultiMamba Implementation**: Proper multimodal input handling
2. **Fixed FusionNet**: Correct parameter interfaces and cross-modal attention
3. **Enhanced Integration**: Proper coupling between temporal, vision, and text features
4. **Optimized Training**: Reduced memory usage and improved stability

## Usage

```python
from models.time_me import TimeMEModel

model = TimeMEModel(config)
predictions = model(time_series_data)
```

## Offline Mode

If running in a restricted network (no external downloads), enable offline mode in your config to avoid remote model fetches and use dummy VLM features:

```python
from configs.config import TimeMEConfig

cfg = TimeMEConfig()
cfg.offline = True  # avoid remote downloads; use zero embeddings for VLM
```

The simplified tests do not require any VLM downloads:

```
python test_model_simple.py
```

For the full model test with real VLM backends, ensure required models are cached locally or network access is available, then run:

```
python test_model.py
```

## Performance

Expected improvements over original Time-VLM:
- Better multimodal fusion through coupled state space models
- Enhanced cross-modal attention mechanisms
- Improved forecasting accuracy on multimodal time series tasks
