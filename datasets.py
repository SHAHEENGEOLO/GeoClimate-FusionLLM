from __future__ import annotations

import torch
from torch.utils.data import Dataset


class WeatherSequenceDataset(Dataset):
    """PyTorch dataset for sequence-to-one weather forecasting."""

    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
