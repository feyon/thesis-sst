"""Smoke test sintetis: validasi pipeline end-to-end TANPA data CMEMS.

Membuat deret SST sintetis (tren musiman + noise), menjalankan
windowing -> latih tiap model 2 epoch -> hitung metrik. Tujuannya hanya
memastikan seluruh komponen tersambung dengan benar.

Jalankan:  python smoke_test.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml

from src.data.dataset import build_loaders
from src.models.factory import build_model
from src.models.persistence import PersistenceModel
from src.training.trainer import train_model, predict
from src.evaluation.metrics import compute_all


def synthetic_series(n_days=4750, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    sst = (28.0
           + 1.5 * np.sin(2 * np.pi * t / 365.25)     # musiman tahunan
           + 0.3 * np.sin(2 * np.pi * t / 30.0)       # intra-musiman
           + rng.normal(0, 0.2, n_days))              # noise
    dates = pd.date_range("2014-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "sst": sst})


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["max_epochs"] = 2          # cepat untuk smoke test
    cfg["training"]["device"] = "cpu"
    cfg["training"]["num_workers"] = 0

    df = synthetic_series()
    lookback, horizon = 14, 3
    bundle = build_loaders(df, cfg, lookback, horizon)
    scaler = bundle["scaler"]
    Xte, yte = bundle["arrays"]["test"]
    y_true = scaler.inverse(yte)
    y_pers = scaler.inverse(PersistenceModel().predict(Xte))

    print(f"Window: lookback={lookback}, horizon={horizon}, n_test={len(yte)}\n")
    for name in ["persistence", "lstm", "transformer", "hybrid"]:
        if name == "persistence":
            y_pred = y_pers
        else:
            model = build_model(name, cfg)
            model, _ = train_model(model, bundle["loaders"], cfg)
            y_pred = scaler.inverse(predict(model, Xte, cfg))
        m = compute_all(y_true, y_pred, y_pers)
        print(f"{name:12s} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
              f"R2={m['r2']:.4f}  NSE={m['nse']:.4f}  SS={m['skill_score']:+.3f}")
    print("\nSmoke test selesai — seluruh komponen tersambung.")


if __name__ == "__main__":
    main()
