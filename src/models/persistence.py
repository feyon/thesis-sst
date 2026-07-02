"""Baseline Persistence: prediksi SST(t+h) = SST(t) (hari terakhir window).

Bukan model neural; tidak ada pelatihan. Dipakai sebagai acuan Skill Score.
"""
import numpy as np


class PersistenceModel:
    """Prediksi = nilai input pada langkah waktu terakhir."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: [n, lookback, 1] -> ambil nilai terakhir tiap window
        return X[:, -1, 0]
