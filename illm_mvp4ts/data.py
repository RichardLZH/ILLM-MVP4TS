"""Chronological multivariate windows. No future target enters any prompt."""
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def read_frame(path, timestamp="date"):
    path = Path(path)
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any() or timestamp not in frame:
        raise ValueError("Missing timestamp column or duplicate variable names.")
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise")
    if frame[timestamp].isna().any() or frame[timestamp].duplicated().any():
        raise ValueError("Timestamps must be nonmissing and unique.")
    return frame.sort_values(timestamp).reset_index(drop=True)


def forecast_frame(path, timestamp="date"):
    frame = read_frame(path, timestamp)
    labels = [name for name in frame.columns if name != timestamp]
    if len(labels) < 2:
        raise ValueError("Multivariate forecasting requires at least two variables.")
    values = frame[labels].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Forecasting inputs must be finite; preprocess missing values without future leakage.")
    intervals = frame[timestamp].diff().dropna()
    if intervals.nunique() != 1 or intervals.iloc[0] <= pd.Timedelta(0):
        raise ValueError("Forecasting requires a regular time grid; preprocess irregular data explicitly.")
    return frame, labels, values


class Windows(Dataset):
    def __init__(self, values, target_index, seq_len, horizon, split, train_fraction=.7, val_fraction=.1):
        if seq_len < 2 or horizon < 1 or split not in {"train", "val", "test"}:
            raise ValueError("Invalid window length, horizon or split.")
        if not (0 < train_fraction < 1 and 0 < val_fraction < 1 - train_fraction):
            raise ValueError("Invalid chronological split fractions.")
        self.values = np.asarray(values, dtype=np.float32)
        self.target_index, self.seq_len, self.horizon = target_index, seq_len, horizon
        n = len(values)
        train_end, val_end = int(n * train_fraction), int(n * (train_fraction + val_fraction))
        start, end = {"train": (seq_len, train_end), "val": (max(seq_len, train_end), val_end),
                      "test": (max(seq_len, val_end), n)}[split]
        self.origins = list(range(start, end - horizon + 1))
        if not self.origins:
            raise ValueError(f"No {split} windows; supply more rows or shorten the window.")

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = self.origins[index]
        return (self.values[origin - self.seq_len:origin].copy(),
                self.values[origin:origin + self.horizon, self.target_index].copy(), origin)
