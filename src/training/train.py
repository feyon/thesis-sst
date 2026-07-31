"""
Training satu eksperimen (satu kombinasi lookback, horizon, model) dari
grid eksperimen yang didefinisikan di config.yaml (`experiment:` section).

Membaca:
  - data.windowed_dir / lb{N:02d}_h{H:02d}/  -> X_train, y_train, X_val, y_val
  - training.*      -> device, batch_size, max_epochs, lr, early stopping
  - model.<nama>.*  -> hyperparameter arsitektur

Cara jalan (lokal, CPU):
    python -m src.training.train --config configs/config.yaml \\
        --lookback 30 --horizon 7 --model lstm

Cara jalan (SLURM + apptainer, GPU):
    cd ~/thesis-sst
    srun --partition=gpu_riset --gres=gpu:1 --cpus-per-task=4 --mem=16G \\
      --time=02:00:00 \\
      apptainer exec --nv --pwd /workspace \\
      --bind ~/thesis-sst/data:/workspace/data \\
      --bind ~/thesis-sst/configs:/workspace/configs \\
      --bind ~/thesis-sst/src:/workspace/src \\
      --bind ~/thesis-sst/results:/workspace/results \\
      ~/thesis-sst/thesis-sst.sif \\
      python -m src.training.train --config configs/config.yaml \\
        --lookback 30 --horizon 7 --model hybrid

Catatan desain (baca sebelum dipakai utk hasil final thesis):
  - Model HYBRID di sini adalah LSTM -> proyeksi linear -> TransformerEncoder
    -> Linear output, sesuai persis parameter di config.model.hybrid
    (lstm_hidden, proj_dim, nhead, num_transformer_layers, dim_feedforward).
    Kalau kamu masih memakai desain HybridLSTMTransformer yang lebih detail
    (BiLSTM Local Temporal Encoder + SpatioTemporal Transformer Encoder +
    Adaptive Fusion Gate) dari diskusi sebelumnya, beri tahu -> arsitektur
    di file ini perlu disesuaikan/diganti dengan implementasi tsb.
  - Semua 6 mooring dilatih SEBAGAI SATU MODEL GABUNGAN (shared weights),
    TANPA location embedding (mooring_id belum dipakai sbg fitur model).
    Ini bisa ditambahkan belakangan kalau hasil awal menunjukkan perlu
    membedakan perilaku antar titik.
  - Loss training: MSE (rata2 di semua langkah horizon). Metrik penuh
    (R2, NSE, Skill Score) dihitung di script evaluasi terpisah, bukan di
    sini -- training.py hanya melaporkan RMSE/MAE ringkas per epoch.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


# ==========================================================================
# Config & util
# ==========================================================================
def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        print("PERINGATAN: device 'cuda' diminta tapi tidak terdeteksi "
              "-> fallback ke CPU. Cek alokasi SLURM (--gres=gpu:1) dan "
              "flag 'apptainer exec --nv'.")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("PERINGATAN: device 'mps' diminta tapi tidak tersedia di "
              "host ini -> fallback ke CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# ==========================================================================
# Dataset
# ==========================================================================
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_split(windowed_dir, lookback, horizon, split_name):
    combo_dir = windowed_dir / f"lb{lookback:02d}_h{horizon:02d}"
    X = np.load(combo_dir / f"X_{split_name}.npy")
    y = np.load(combo_dir / f"y_{split_name}.npy")
    return X, y


# ==========================================================================
# Model: LSTM
# ==========================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, n_out, hidden_size=50, num_layers=1,
                 dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, n_out)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last = out[:, -1, :]           # (B, hidden_size)
        return self.head(last)          # (B, n_out)


# ==========================================================================
# Model: Transformer (encoder-only)
# ==========================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, input_size, n_out, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x):
        z = self.input_proj(x)
        z = self.pos_enc(z)
        z = self.encoder(z)
        last = z[:, -1, :]
        return self.head(last)


# ==========================================================================
# Model: Hybrid (LSTM feature extractor -> proyeksi -> Transformer encoder)
# ==========================================================================
class HybridModel(nn.Module):
    def __init__(self, input_size, n_out, lstm_hidden=50, proj_dim=52,
                 nhead=4, num_transformer_layers=2, dim_feedforward=128,
                 dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True,
        )
        self.proj = nn.Linear(lstm_hidden, proj_dim)
        self.pos_enc = PositionalEncoding(proj_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_transformer_layers
        )
        self.head = nn.Linear(proj_dim, n_out)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)      # (B, T, lstm_hidden)
        z = self.proj(lstm_out)          # (B, T, proj_dim)
        z = self.pos_enc(z)
        z = self.encoder(z)
        last = z[:, -1, :]
        return self.head(last)


def build_model(model_name, input_size, n_out, model_cfg):
    if model_name == "lstm":
        c = model_cfg["lstm"]
        return LSTMModel(input_size, n_out, hidden_size=c["hidden_size"],
                         num_layers=c["num_layers"], dropout=c["dropout"])
    elif model_name == "transformer":
        c = model_cfg["transformer"]
        return TransformerModel(input_size, n_out, d_model=c["d_model"],
                                nhead=c["nhead"], num_layers=c["num_layers"],
                                dim_feedforward=c["dim_feedforward"],
                                dropout=c["dropout"])
    elif model_name == "hybrid":
        c = model_cfg["hybrid"]
        return HybridModel(input_size, n_out, lstm_hidden=c["lstm_hidden"],
                           proj_dim=c["proj_dim"], nhead=c["nhead"],
                           num_transformer_layers=c["num_transformer_layers"],
                           dim_feedforward=c["dim_feedforward"],
                           dropout=c["dropout"])
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")


# ==========================================================================
# Training loop
# ==========================================================================
def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_mae, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = X.size(0)
            total_loss += loss.item() * bs
            total_mae += torch.abs(pred - y).mean().item() * bs
            n += bs

    return total_loss / n, total_mae / n  # mean MSE, mean MAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--lookback", type=int, required=True,
                        help="Panjang window input (hari), mis. 7/14/21/30")
    parser.add_argument("--horizon", type=int, required=True,
                        help="Horizon prediksi (hari), mis. 1/3/7/14")
    parser.add_argument("--model", type=str, required=True,
                        choices=["lstm", "transformer", "hybrid"])
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override training.max_epochs dari config (opsional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["random_seed"])

    train_cfg = cfg["training"]
    device = resolve_device(train_cfg["device"])
    print(f"Device dipakai: {device}")

    windowed_dir = REPO_ROOT / cfg["data"]["windowed_dir"]
    results_dir = REPO_ROOT / cfg["evaluation"]["results_dir"]
    run_name = f"{args.model}_lb{args.lookback:02d}_h{args.horizon:02d}"
    run_dir = results_dir / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Eksperimen: model={args.model}, lookback={args.lookback}, "
          f"horizon={args.horizon} ===")

    # ------------------------------------------------------------------
    print("Memuat data window...")
    X_train, y_train = load_split(windowed_dir, args.lookback, args.horizon, "train")
    X_val, y_val = load_split(windowed_dir, args.lookback, args.horizon, "val")
    print(f"  X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  X_val   {X_val.shape}, y_val   {y_val.shape}")

    input_size = X_train.shape[-1]
    n_out = y_train.shape[-1]
    print(f"  input_size (F) = {input_size}, n_out (horizon) = {n_out}")

    train_ds = WindowDataset(X_train, y_train)
    val_ds = WindowDataset(X_val, y_val)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=train_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=train_cfg["pin_memory"],
    )

    # ------------------------------------------------------------------
    model = build_model(args.model, input_size, n_out, cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model '{args.model}' dibangun, {n_params:,} parameter trainable")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    max_epochs = args.max_epochs or train_cfg["max_epochs"]
    patience = train_cfg["early_stopping_patience"]

    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    print(f"\nMulai training (max_epochs={max_epochs}, "
          f"early_stopping_patience={patience})...")
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        train_mse, train_mae = run_epoch(model, train_loader, criterion,
                                         device, optimizer)
        val_mse, val_mae = run_epoch(model, val_loader, criterion, device)
        train_rmse, val_rmse = train_mse ** 0.5, val_mse ** 0.5

        history.append({
            "epoch": epoch, "train_rmse": train_rmse, "train_mae": train_mae,
            "val_rmse": val_rmse, "val_mae": val_mae,
        })
        print(f"  Epoch {epoch:3d}/{max_epochs} | "
              f"train RMSE {train_rmse:.4f} MAE {train_mae:.4f} | "
              f"val RMSE {val_rmse:.4f} MAE {val_mae:.4f}")

        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_rmse": val_rmse,
                "config_model": args.model,
                "lookback": args.lookback,
                "horizon": args.horizon,
                "input_size": input_size,
            }, run_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping di epoch {epoch} "
                     f"(tidak membaik {patience} epoch berturut-turut, "
                     f"terbaik epoch {best_epoch} val RMSE {best_val_rmse:.4f})")
                break

    elapsed = time.time() - t0
    print(f"\nSelesai dalam {elapsed:.1f} detik. "
          f"Best val RMSE = {best_val_rmse:.4f} @ epoch {best_epoch}")

    # ------------------------------------------------------------------
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(run_dir / "history.csv", index=False)

    summary = {
        "run_name": run_name,
        "model": args.model,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "input_size": input_size,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "total_epochs_run": len(history),
        "elapsed_sec": round(elapsed, 1),
        "device": str(device),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Checkpoint & log tersimpan di: {run_dir}/")
    print("  - best_model.pt  (state_dict model terbaik)")
    print("  - history.csv    (RMSE/MAE per epoch)")
    print("  - summary.json   (ringkasan run)")


if __name__ == "__main__":
    main()