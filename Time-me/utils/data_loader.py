import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting"""
    def __init__(self, root_path, data, seq_len, pred_len, split='train', scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split

        # Load data
        data_path = os.path.join(root_path, f'{data}.csv')
        self.data_df = pd.read_csv(data_path)

        # Extract features and targets
        self.features = self.data_df.iloc[:, :-1].values  # All columns except last
        self.targets = self.data_df.iloc[:, -1].values     # Last column as target

        # Normalize data
        if scale:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = None

        # Create sequences
        self.sequences = self._create_sequences()

        # Split data
        self._split_data()

    def _create_sequences(self):
        """Create input-output sequences"""
        sequences = []
        total_len = len(self.features)

        for i in range(total_len - self.seq_len - self.pred_len + 1):
            # Input sequence
            x = self.features[i:i + self.seq_len]
            # Target sequence
            y = self.targets[i + self.seq_len:i + self.seq_len + self.pred_len]
            sequences.append((x, y))

        return sequences

    def _split_data(self):
        """Split data into train/validation/test sets"""
        total_sequences = len(self.sequences)

        if self.split == 'train':
            start_idx = 0
            end_idx = int(0.7 * total_sequences)
        elif self.split == 'val':
            start_idx = int(0.7 * total_sequences)
            end_idx = int(0.85 * total_sequences)
        elif self.split == 'test':
            start_idx = int(0.85 * total_sequences)
            end_idx = total_sequences
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.sequences = self.sequences[start_idx:end_idx]

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
    def __init__(self, root_path, data, seq_len, pred_len, split='train', scale=True):
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

        # Normalize data
        if scale:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = None

        # Create sequences
        self.sequences = self._create_sequences()

        # Split data
        self._split_data()

    def _create_sequences(self):
        """Create input-output sequences for multivariate data"""
        sequences = []
        total_len, n_features = self.data.shape

        for i in range(total_len - self.seq_len - self.pred_len + 1):
            # Input sequence [seq_len, n_features]
            x = self.data[i:i + self.seq_len, :]
            # Target sequence [pred_len, n_features]
            y = self.data[i + self.seq_len:i + self.seq_len + self.pred_len, :]
            sequences.append((x, y))

        return sequences

    def _split_data(self):
        """Split data into train/validation/test sets"""
        total_sequences = len(self.sequences)

        if self.split == 'train':
            start_idx = 0
            end_idx = int(0.7 * total_sequences)
        elif self.split == 'val':
            start_idx = int(0.7 * total_sequences)
            end_idx = int(0.85 * total_sequences)
        elif self.split == 'test':
            start_idx = int(0.85 * total_sequences)
            end_idx = total_sequences
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.sequences = self.sequences[start_idx:end_idx]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]

        # Convert to tensors and reshape for model [batch, seq_len, features]
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)

        return x_tensor, y_tensor


def create_data_loaders(root_path, data, seq_len, pred_len, batch_size=32):
    """Create train, validation, and test data loaders"""
    # Create datasets
    train_dataset = MultivariateTimeSeriesDataset(
        root_path=root_path,
        data=data,
        seq_len=seq_len,
        pred_len=pred_len,
        split='train'
    )

    val_dataset = MultivariateTimeSeriesDataset(
        root_path=root_path,
        data=data,
        seq_len=seq_len,
        pred_len=pred_len,
        split='val'
    )

    test_dataset = MultivariateTimeSeriesDataset(
        root_path=root_path,
        data=data,
        seq_len=seq_len,
        pred_len=pred_len,
        split='test'
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

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
