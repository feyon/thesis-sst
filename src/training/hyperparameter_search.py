"""Pencarian eksperimen penuh: 4 lookback x 4 horizon x 4 model = 64 run.

Untuk tiap kombinasi:
  - Persistence  : tanpa pelatihan, langsung hitung metrik.
  - LSTM/Transformer/Hybrid : latih + early stopping, evaluasi di test set.
Semua metrik dihitung pada skala asli (°C, via inverse z-score).
Hasil disimpan ke results/results_all.csv

Jalankan: python -m src.training.hyperparameter_search --config configs/config.yaml
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from ..data.dataset import build_loaders
from ..models.factory import build_model
from ..models.persistence import PersistenceModel
from ..training.trainer import train_model, predict
from ..evaluation.metrics import compute_all


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _persistence_pred(X: np.ndarray) -> np.ndarray:
    return PersistenceModel().predict(X)


def run(cfg: dict, data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["date"])
    exp = cfg["experiment"]
    rows = []

    for lookback in exp["lookback_windows"]:
        for horizon in exp["horizons"]:
            bundle = build_loaders(df, cfg, lookback, horizon)
            scaler = bundle["scaler"]
            Xte, yte = bundle["arrays"]["test"]
            if len(yte) == 0:
                continue

            # target & persistence pada skala asli
            y_true = scaler.inverse(yte)
            y_pers = scaler.inverse(_persistence_pred(Xte))

            for model_name in exp["models"]:
                t0 = time.time()
                if model_name == "persistence":
                    y_pred = y_pers
                else:
                    model = build_model(model_name, cfg)
                    model, _ = train_model(model, bundle["loaders"], cfg)
                    y_pred = scaler.inverse(predict(model, Xte, cfg))

                m = compute_all(y_true, y_pred, y_pers)
                m.update(dict(model=model_name, lookback=lookback,
                              horizon=horizon, n_test=len(yte),
                              seconds=round(time.time() - t0, 1)))
                rows.append(m)
                print(f"[{model_name:11s}] N={lookback:2d} h={horizon:2d} "
                      f"RMSE={m['rmse']:.4f} SS={m['skill_score']:+.3f} "
                      f"({m['seconds']}s)")

    res = pd.DataFrame(rows)
    out = Path(cfg["evaluation"]["results_dir"]); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results_all.csv", index=False)
    print("\nTersimpan:", out / "results_all.csv")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="data/processed/sst_series.csv")
    cfg = load_config(p.parse_args().config)
    run(cfg, p.parse_args().data)
