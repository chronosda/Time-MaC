import math

import torch
import torch.nn as nn


DATASET_NAMES = (
    'ETTh1',
    'ETTh2',
    'ETTm1',
    'ETTm2',
    'electricity',
    'weather',
    'traffic',
)
DATASET_TO_ID = {name.lower(): index for index, name in enumerate(DATASET_NAMES)}


def resolve_dataset_id(dataset_name):
    """Map one of the seven main benchmark names to a stable embedding index."""
    normalized = str(dataset_name).strip().lower()
    if normalized.endswith('.csv'):
        normalized = normalized[:-4]
    if normalized not in DATASET_TO_ID:
        supported = ', '.join(DATASET_NAMES)
        raise ValueError(
            f"Structured context requires one of the seven main datasets: {supported}; "
            f"got {dataset_name!r}"
        )
    return DATASET_TO_ID[normalized]


class StructuredContextEncoder(nn.Module):
    """Encode dataset identity and normalized numeric context without language."""

    numeric_feature_count = 6

    def __init__(self, config):
        super().__init__()
        self.dataset_name = getattr(config, 'data', '')
        self.dataset_embedding_dim = int(
            getattr(config, 'dataset_embedding_dim', 32)
        )
        self.hidden_dim = int(getattr(config, 'context_hidden_dim', 128))
        self.output_dim = int(getattr(config, 'context_output_dim', 256))
        self.length_scale = float(getattr(config, 'context_length_scale', 1024.0))
        self.pred_len = int(getattr(config, 'pred_len', 96))

        if min(self.dataset_embedding_dim, self.hidden_dim, self.output_dim) <= 0:
            raise ValueError("Structured context dimensions must be positive")
        if self.length_scale <= 1:
            raise ValueError("context_length_scale must be greater than 1")

        self.dataset_embedding = nn.Embedding(
            len(DATASET_NAMES),
            self.dataset_embedding_dim,
        )
        input_dim = self.dataset_embedding_dim + self.numeric_feature_count
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def _numeric_features(self, x):
        batch_size, seq_len, _ = x.shape
        flattened = x.reshape(batch_size, -1)

        # TimeMEModel supplies per-variable standardized inputs. Tanh bounds the
        # remaining sample-level statistics and makes outliers less dominant.
        minimum = torch.tanh(flattened.amin(dim=1))
        maximum = torch.tanh(flattened.amax(dim=1))
        median = torch.tanh(flattened.median(dim=1).values)
        if seq_len > 1:
            trend = x.diff(dim=1).mean(dim=(1, 2))
        else:
            trend = x.new_zeros(batch_size)
        trend = torch.tanh(trend)

        log_scale = math.log1p(self.length_scale)
        seq_feature = min(math.log1p(seq_len) / log_scale, 1.0)
        pred_feature = min(math.log1p(self.pred_len) / log_scale, 1.0)
        seq_values = x.new_full((batch_size,), seq_feature)
        pred_values = x.new_full((batch_size,), pred_feature)

        return torch.stack(
            [maximum, minimum, median, trend, seq_values, pred_values],
            dim=-1,
        )

    def forward(self, x, dataset_name=None):
        if x.ndim != 3:
            raise ValueError(
                f"Structured context expects [batch, time, variables], got {tuple(x.shape)}"
            )
        resolved_name = self.dataset_name if dataset_name is None else dataset_name
        dataset_id = resolve_dataset_id(resolved_name)
        ids = torch.full(
            (x.shape[0],),
            dataset_id,
            dtype=torch.long,
            device=x.device,
        )
        dataset_features = self.dataset_embedding(ids)
        numeric_features = self._numeric_features(x)
        return self.mlp(torch.cat([dataset_features, numeric_features], dim=-1))

    def get_model_info(self):
        return {
            'dataset_embedding_dim': self.dataset_embedding_dim,
            'numeric_feature_count': self.numeric_feature_count,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
