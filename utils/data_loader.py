import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _resolve_split_lengths(total_len, train_ratio, val_ratio, calib_ratio, test_ratio):
    ratios = np.array([train_ratio, val_ratio, calib_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("Split ratios must be non-negative.")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratios.sum():.6f} "
            f"(train={train_ratio}, val={val_ratio}, calib={calib_ratio}, test={test_ratio})."
        )

    raw_lengths = ratios * total_len
    lengths = np.floor(raw_lengths).astype(int)
    remainder = total_len - int(lengths.sum())
    if remainder > 0:
        order = np.argsort(-(raw_lengths - lengths))
        for idx in order[:remainder]:
            lengths[idx] += 1
    return lengths.tolist()


def _split_boundaries(total_len, train_ratio, val_ratio, calib_ratio, test_ratio):
    train_len, val_len, calib_len, test_len = _resolve_split_lengths(
        total_len, train_ratio, val_ratio, calib_ratio, test_ratio
    )
    train_end = train_len
    val_end = train_end + val_len
    calib_end = val_end + calib_len
    test_end = calib_end + test_len
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "calib": (val_end, calib_end),
        "test": (calib_end, test_end),
    }


def _ett_standard_borders(data_name, seq_len, split):
    """Standard ETT benchmark borders used by Informer/Autoformer-style loaders."""
    name = data_name.lower()
    if name.startswith("ettm"):
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = val_end + 4 * 30 * 24 * 4
    elif name.startswith("etth"):
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
    else:
        raise ValueError("ett_standard split is only defined for ETTh*/ETTm* datasets.")

    borders = {
        "train": (0, train_end),
        "val": (train_end - seq_len, val_end),
        "calib": (val_end - seq_len, test_end),
        "test": (val_end - seq_len, test_end),
    }
    if split not in borders:
        raise ValueError(f"Unknown split: {split}")
    return borders[split], (0, train_end)


def _custom_standard_borders(total_len, seq_len, split):
    """Standard custom-dataset borders used by Time-Series-Library loaders.

    Public long-term forecasting benchmarks such as Electricity, Weather, and
    Traffic commonly use 70% train, 10% validation, and 20% test. Validation
    and test include a seq_len lookback overlap from the previous segment.
    """
    num_train = int(total_len * 0.7)
    num_test = int(total_len * 0.2)
    num_val = total_len - num_train - num_test

    train_end = num_train
    val_end = num_train + num_val
    test_start = total_len - num_test

    borders = {
        "train": (0, train_end),
        "val": (train_end - seq_len, val_end),
        "calib": (train_end - seq_len, val_end),
        "test": (test_start - seq_len, total_len),
    }
    if split not in borders:
        raise ValueError(f"Unknown split: {split}")
    return borders[split], (0, train_end)


def _make_split_sequences(data, seq_len, pred_len, start_idx, end_idx):
    split_data = data[start_idx:end_idx]
    num_windows = len(split_data) - seq_len - pred_len + 1
    if num_windows <= 0:
        return []

    sequences = []
    for i in range(num_windows):
        x = split_data[i:i + seq_len]
        y = split_data[i + seq_len:i + seq_len + pred_len]
        sequences.append((x, y))
    return sequences


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting"""
    def __init__(
        self,
        root_path,
        data,
        seq_len,
        pred_len,
        split='train',
        scale=True,
        train_ratio=0.7,
        val_ratio=0.1,
        calib_ratio=0.1,
        test_ratio=0.1,
        data_split='ratio',
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split

        # Load data
        data_path = os.path.join(root_path, f'{data}.csv')
        self.data_df = pd.read_csv(data_path)

        # Extract features and targets
        self.features = self.data_df.iloc[:, :-1].values  # All columns except last
        self.targets = self.data_df.iloc[:, -1].values     # Last column as target

        if data_split == 'ett_standard':
            (start_idx, end_idx), (train_start, train_end) = _ett_standard_borders(
                data, seq_len, split
            )
            boundaries = {split: (start_idx, end_idx)}
        elif data_split == 'custom_standard':
            (start_idx, end_idx), (train_start, train_end) = _custom_standard_borders(
                len(self.features), seq_len, split
            )
            boundaries = {split: (start_idx, end_idx)}
        else:
            boundaries = _split_boundaries(
                len(self.features), train_ratio, val_ratio, calib_ratio, test_ratio
            )
            train_start, train_end = boundaries['train']

        # Normalize data using train split only
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(self.features[train_start:train_end])
            self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None

        # Split data
        self._split_data(boundaries)

    def _split_data(self, boundaries):
        """Split raw timeline before sequence creation to avoid cross-split leakage."""
        if self.split not in boundaries:
            raise ValueError(f"Unknown split: {self.split}")

        start_idx, end_idx = boundaries[self.split]
        split_features = self.features[start_idx:end_idx]
        split_targets = self.targets[start_idx:end_idx]

        num_windows = len(split_features) - self.seq_len - self.pred_len + 1
        self.sequences = []
        for i in range(max(num_windows, 0)):
            x = split_features[i:i + self.seq_len]
            y = split_targets[i + self.seq_len:i + self.seq_len + self.pred_len]
            self.sequences.append((x, y))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]

        # Convert to tensors
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)

        return x_tensor, y_tensor


class MultivariateTimeSeriesDataset(Dataset):
    """Dataset for multivariate time series forecasting"""
    def __init__(
        self,
        root_path,
        data,
        seq_len,
        pred_len,
        split='train',
        scale=True,
        train_ratio=0.7,
        val_ratio=0.1,
        calib_ratio=0.1,
        test_ratio=0.1,
        data_split='ratio',
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split

        # Load data
        data_path = os.path.join(root_path, f'{data}.csv')
        self.data_df = pd.read_csv(data_path)

        # Handle time columns and keep numeric features only
        df = self.data_df.copy()
        # Drop obvious time columns if present
        time_cols = [c for c in df.columns if c.lower() in ("date", "datetime", "timestamp", "time")]
        for c in time_cols:
            try:
                df[c] = pd.to_datetime(df[c])
                df = df.drop(columns=[c])
            except Exception:
                df = df.drop(columns=[c])
        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])
        if df.shape[1] == 0:
            raise ValueError("No numeric feature columns found after removing time columns.")
        self.data = df.values

        if data_split == 'ett_standard':
            (start_idx, end_idx), (train_start, train_end) = _ett_standard_borders(
                data, seq_len, split
            )
            boundaries = {split: (start_idx, end_idx)}
        elif data_split == 'custom_standard':
            (start_idx, end_idx), (train_start, train_end) = _custom_standard_borders(
                len(self.data), seq_len, split
            )
            boundaries = {split: (start_idx, end_idx)}
        else:
            boundaries = _split_boundaries(
                len(self.data), train_ratio, val_ratio, calib_ratio, test_ratio
            )
            train_start, train_end = boundaries['train']

        # Normalize data using train split only
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(self.data[train_start:train_end])
            self.data = self.scaler.transform(self.data)
        else:
            self.scaler = None

        # Split data
        self._split_data(boundaries)

    def _split_data(self, boundaries):
        """Split raw timeline before sequence creation to avoid cross-split leakage."""
        if self.split not in boundaries:
            raise ValueError(f"Unknown split: {self.split}")

        start_idx, end_idx = boundaries[self.split]
        self.sequences = _make_split_sequences(
            self.data, self.seq_len, self.pred_len, start_idx, end_idx
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]

        # Convert to tensors and reshape for model [batch, seq_len, features]
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)

        return x_tensor, y_tensor


def create_data_loaders(
    root_path,
    data,
    seq_len,
    pred_len,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.1,
    calib_ratio=0.1,
    test_ratio=0.1,
    data_split='ratio',
):
    """Create train, validation, calibration, and test data loaders."""
    # Create datasets
    train_dataset = MultivariateTimeSeriesDataset(
        root_path=root_path,
        data=data,
        seq_len=seq_len,
        pred_len=pred_len,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        calib_ratio=calib_ratio,
        test_ratio=test_ratio,
        data_split=data_split,
    )

    val_dataset = MultivariateTimeSeriesDataset(
        root_path=root_path,
        data=data,
        seq_len=seq_len,
        pred_len=pred_len,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        calib_ratio=calib_ratio,
        test_ratio=test_ratio,
        data_split=data_split,
    )

    calib_dataset = MultivariateTimeSeriesDataset(
        root_path=root_path,
        data=data,
        seq_len=seq_len,
        pred_len=pred_len,
        split='calib',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        calib_ratio=calib_ratio,
        test_ratio=test_ratio,
        data_split=data_split,
    )

    test_dataset = MultivariateTimeSeriesDataset(
        root_path=root_path,
        data=data,
        seq_len=seq_len,
        pred_len=pred_len,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        calib_ratio=calib_ratio,
        test_ratio=test_ratio,
        data_split=data_split,
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, calib_loader, test_loader
