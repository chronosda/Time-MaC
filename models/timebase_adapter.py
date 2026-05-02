import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class TimeBaseConfig:
    """Minimal config wrapper matching TimeBase.Model expectations."""

    # Core sequence settings
    seq_len: int = 720
    pred_len: int = 96
    enc_in: int = 7

    # TimeBase-specific hyperparameters
    period_len: int = 24
    basis_num: int = 6
    individual: bool = False

    # Normalization and regularization
    use_period_norm: bool = True
    use_orthogonal: bool = True


def _import_timebase_model() -> type:
    """
    Dynamically import TimeBase.Model from the upstream TimeBase repo.

    By default this uses /home/chronos/TimeBase, but you can override the
    path by setting the TIMEBASE_ROOT environment variable.
    """
    timebase_root = os.environ.get("TIMEBASE_ROOT", "/home/chronos/TimeBase")
    root_path = Path(timebase_root).expanduser().resolve()
    models_dir = root_path / "models"

    if not models_dir.exists():
        raise RuntimeError(
            f"TimeBase models directory not found at {models_dir}. "
            f"Set TIMEBASE_ROOT to your TimeBase repo path."
        )

    try:
        # Avoid clashing with this project's own `models` package by
        # importing directly from the TimeBase `models` directory.
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))
        from TimeBase import Model as TimeBaseModel  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Failed to import TimeBase.Model from TimeBase repository. "
            "Ensure the TimeBase repo is available and importable."
        ) from exc

    return TimeBaseModel


class TimeBaseForecaster(nn.Module):
    """
    Thin adapter around TimeBase.Model to make it plug-and-play
    inside the Time-me training pipeline.

    Input/Output:
      - expects x: [batch, seq_len, channels]
      - returns y: [batch, pred_len, channels]
    """

    def __init__(
        self,
        config: Optional[TimeBaseConfig] = None,
        lambda_orth: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config or TimeBaseConfig()
        self.lambda_orth = float(lambda_orth)

        timebase_model_cls = _import_timebase_model()
        # Under the hood TimeBase.Model reads attributes directly
        self.backbone = timebase_model_cls(self.config)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward through TimeBase.

        Depending on use_orthogonal, TimeBase.Model either returns:
          - y                            (use_orthogonal == False)
          - (y, orthogonal_loss_term)    (use_orthogonal == True)
        """
        out = self.backbone(x)

        if self.config.use_orthogonal and isinstance(out, tuple):
            preds, orth_loss = out
            # Surface orthogonal loss scaled by lambda_orth so callers can add it
            return preds, self.lambda_orth * orth_loss

        # Normal case: just predictions
        return out


def build_timebase_for_dataset(
    seq_len: int,
    pred_len: int,
    enc_in: int,
    period_len: int = 24,
    basis_num: int = 6,
    lambda_orth: float = 0.0,
) -> TimeBaseForecaster:
    """
    Convenience constructor for ETTh1-style datasets.

    Example:
        model = build_timebase_for_dataset(
            seq_len=720, pred_len=96, enc_in=7, period_len=24, basis_num=6
        )
    """
    cfg = TimeBaseConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        period_len=period_len,
        basis_num=basis_num,
        individual=False,
        use_period_norm=True,
        use_orthogonal=lambda_orth > 0.0,
    )
    return TimeBaseForecaster(config=cfg, lambda_orth=lambda_orth)
