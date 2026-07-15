"""Dataset & DataLoader untuk SST univariat.

Kunci metodologis:
  * Normalisasi z-score DI-FIT HANYA PADA SET PELATIHAN (anti-leakage),
    lalu statistik (mean, std) diterapkan ke val & test.
  * Sliding window dibentuk DI DALAM tiap split (input & target satu split),
    sehingga tidak ada kebocoran antar-split.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class Scaler:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def fit_scaler(train_values: np.ndarray) -> Scaler:
    return Scaler(mean=float(np.nanmean(train_values)),
                  std=float(np.nanstd(train_values) + 1e-8))


def _slice_split(df: pd.DataFrame, span) -> pd.DataFrame:
    start, end = pd.Timestamp(span[0]), pd.Timestamp(span[1])
    m = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[m].reset_index(drop=True)


def make_windows(values: np.ndarray, lookback: int, horizon: int):
    """values: array 1D ternormalisasi. Return X[n,lookback,1], y[n]."""
    X, y = [], []
    last = len(values) - horizon
    for t in range(lookback, last + 1):
        X.append(values[t - lookback:t])
        y.append(values[t + horizon - 1])
    if not X:
        return (np.empty((0, lookback, 1), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    X = np.asarray(X, dtype=np.float32)[..., None]   # -> [n, lookback, 1]
    y = np.asarray(y, dtype=np.float32)
    return X, y


class SSTWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def build_loaders(df: pd.DataFrame, cfg: dict, lookback: int, horizon: int):
    """Kembalikan dict berisi loader train/val/test + scaler + persistence target.

    Persistence baseline = nilai SST pada hari terakhir window (input[-1]),
    dipakai untuk Skill Score.
    """
    sp, tr = cfg["split"], cfg["training"]

    df_tr = _slice_split(df, sp["train"])
    df_va = _slice_split(df, sp["val"])
    df_te = _slice_split(df, sp["test"])

    scaler = fit_scaler(df_tr["sst"].values)

    def prep(d):
        vals = scaler.transform(d["sst"].values.astype(np.float32))
        return make_windows(vals, lookback, horizon)

    Xtr, ytr = prep(df_tr)
    Xva, yva = prep(df_va)
    Xte, yte = prep(df_te)

    common = dict(batch_size=tr["batch_size"], num_workers=tr["num_workers"],
                  pin_memory=tr["pin_memory"])
    loaders = {
        "train": DataLoader(SSTWindowDataset(Xtr, ytr), shuffle=True, **common),
        "val":   DataLoader(SSTWindowDataset(Xva, yva), shuffle=False, **common),
        "test":  DataLoader(SSTWindowDataset(Xte, yte), shuffle=False, **common),
    }
    return {"loaders": loaders, "scaler": scaler,
            "arrays": {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}}
