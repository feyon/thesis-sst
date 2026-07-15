"""Loop pelatihan: Adam + MSE loss + early stopping (patience).

Mendukung backend Apple Silicon (MPS), CUDA, atau CPU.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn


def get_device(pref: str = "mps") -> torch.device:
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, loaders, cfg, verbose: bool = False):
    tr = cfg["training"]
    device = get_device(tr["device"])
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=tr["learning_rate"])
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience, bad = tr["early_stopping_patience"], 0
    history = {"train": [], "val": []}

    for epoch in range(tr["max_epochs"]):
        # ---- train ----
        model.train()
        tloss = 0.0
        for X, y in loaders["train"]:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tloss += loss.item() * len(y)
        tloss /= max(len(loaders["train"].dataset), 1)

        # ---- validate ----
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for X, y in loaders["val"]:
                X, y = X.to(device), y.to(device)
                vloss += loss_fn(model(X), y).item() * len(y)
        vloss /= max(len(loaders["val"].dataset), 1)

        history["train"].append(tloss)
        history["val"].append(vloss)
        if verbose:
            print(f"  epoch {epoch+1:3d} | train {tloss:.5f} | val {vloss:.5f}")

        # ---- early stopping ----
        if vloss < best_val - 1e-6:
            best_val, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop @ epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, X_array, cfg) -> np.ndarray:
    """Prediksi batch tunggal dari numpy array [n, lookback, 1]."""
    device = get_device(cfg["training"]["device"])
    model = model.to(device).eval()
    X = torch.from_numpy(X_array).to(device)
    return model(X).cpu().numpy()
