"""Metrik evaluasi prediksi SST (dihitung pada skala ASLI / °C)."""
from __future__ import annotations
import numpy as np


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def nse(y, yhat):
    """Nash-Sutcliffe Efficiency (identik bentuk dengan R2 terhadap mean)."""
    denom = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1.0 - np.sum((y - yhat) ** 2) / denom)


def skill_score(y, yhat, y_persistence):
    """SS = 1 - MSE_model / MSE_persistence. Gate utama tesis (> 0.3)."""
    mse_model = np.mean((y - yhat) ** 2)
    mse_pers = np.mean((y - y_persistence) ** 2) + 1e-12
    return float(1.0 - mse_model / mse_pers)


def compute_all(y, yhat, y_persistence):
    return {
        "rmse": rmse(y, yhat),
        "mae": mae(y, yhat),
        "r2": r2(y, yhat),
        "nse": nse(y, yhat),
        "skill_score": skill_score(y, yhat, y_persistence),
    }
