"""Entry point pelatihan model SST.

Mengorkestrasi seluruh pipeline:
  1. Muat sst_series.csv hasil preprocessing
  2. Bentuk sliding window DataLoader (build_loaders)
  3. Instansiasi model via factory
  4. Latih model dengan early stopping (trainer)
  5. Evaluasi pada test set
  6. Simpan checkpoint + hasil metrik ke JSON

Cara menjalankan:
  python -m src.models.train \\
      --model hybrid \\
      --lookback 14 \\
      --horizon 1 \\
      --device cuda \\
      --config configs/config.yaml \\
      --checkpoint-dir models/checkpoints
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import build_loaders
from src.models.factory import build_model
from src.training.trainer import get_device, train_model, predict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_persist: np.ndarray) -> dict:
    """Hitung RMSE, MAE, MAPE, R², NSE, Skill Score."""
    eps = 1e-8
    n = len(y_true)

    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # MAPE (hindari div/0)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + eps))

    # NSE (Nash-Sutcliffe Efficiency)
    nse = float(1 - ss_res / (ss_tot + eps))   # identik R² untuk regresi

    # Skill Score vs Persistence baseline
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_persist = np.mean((y_true - y_persist) ** 2)
    skill_score = float(1 - mse_model / (mse_persist + eps))

    return {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape": round(mape, 4),
        "r2": round(r2, 6),
        "nse": round(nse, 6),
        "skill_score": round(skill_score, 6),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train SST prediction model (LSTM / Transformer / Hybrid)"
    )
    parser.add_argument("--model", required=True,
                        choices=["lstm", "transformer", "hybrid"],
                        help="Arsitektur model yang akan dilatih")
    parser.add_argument("--lookback", type=int, required=True,
                        choices=[7, 14, 21, 30],
                        help="Panjang window input (hari)")
    parser.add_argument("--horizon", type=int, required=True,
                        choices=[1, 3, 7, 14],
                        help="Horizon prediksi (hari ke depan)")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path ke file konfigurasi YAML")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints",
                        help="Direktori penyimpanan checkpoint model")
    parser.add_argument("--device", default=None,
                        help="Override device: cuda | mps | cpu")
    parser.add_argument("--verbose", action="store_true",
                        help="Tampilkan loss per epoch")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override device dari argumen CLI jika diberikan
    if args.device:
        cfg["training"]["device"] = args.device

    run_id = f"{args.model}_lb{args.lookback}_h{args.horizon}"
    print(f"\n{'='*60}")
    print(f"  Run   : {run_id}")
    print(f"  Device: {cfg['training']['device']}")
    print(f"{'='*60}\n")

    # ── 1. Muat data ──────────────────────────────────────────────────────────
    csv_path = Path(cfg["data"]["processed_dir"]) / "sst_series.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} tidak ditemukan. "
            "Jalankan preprocess.py terlebih dahulu."
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"Data SST dimuat: {len(df)} hari "
          f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")

    # ── 2. Bangun DataLoader ──────────────────────────────────────────────────
    data = build_loaders(df, cfg, lookback=args.lookback, horizon=args.horizon)
    loaders = data["loaders"]
    scaler  = data["scaler"]
    Xte, yte = data["arrays"]["test"]

    print(f"Window train : {len(loaders['train'].dataset):,}")
    print(f"Window val   : {len(loaders['val'].dataset):,}")
    print(f"Window test  : {len(loaders['test'].dataset):,}\n")

    # ── 3. Bangun model ───────────────────────────────────────────────────────
    model = build_model(args.model, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model        : {args.model} ({n_params:,} parameter)\n")

    # ── 4. Training ───────────────────────────────────────────────────────────
    t0 = time.time()
    model, history = train_model(
        model, loaders, cfg, verbose=args.verbose
    )
    elapsed = time.time() - t0
    print(f"\nTraining selesai: {elapsed:.1f} detik "
          f"({len(history['train'])} epoch)")
    print(f"  Best val loss : {min(history['val']):.6f}")

    # ── 5. Evaluasi test set ──────────────────────────────────────────────────
    # Prediksi model
    y_pred_norm = predict(model, Xte, cfg)

    # Persistence baseline: nilai SST terakhir dalam window (input[-1, 0])
    y_persist_norm = Xte[:, -1, 0]

    # Inverse transform ke skala °C asli
    y_true  = scaler.inverse(yte)
    y_pred  = scaler.inverse(y_pred_norm)
    y_persist = scaler.inverse(y_persist_norm)

    metrics = compute_metrics(y_true, y_pred, y_persist)
    print(f"\nMetrik Test Set:")
    print(f"  RMSE        : {metrics['rmse']:.4f} °C")
    print(f"  MAE         : {metrics['mae']:.4f} °C")
    print(f"  MAPE        : {metrics['mape']:.2f} %")
    print(f"  R²          : {metrics['r2']:.4f}")
    print(f"  NSE         : {metrics['nse']:.4f}")
    print(f"  Skill Score : {metrics['skill_score']:.4f} "
          f"({'✓ > 0.3' if metrics['skill_score'] > 0.3 else '✗ < 0.3'})")

    # ── 6. Simpan hasil ───────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint model weights
    ckpt_path = ckpt_dir / f"{run_id}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": args.model,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "history": history,
        "metrics": metrics,
        "config": cfg,
    }, ckpt_path)
    print(f"\nCheckpoint   : {ckpt_path}")

    # Hasil metrik ke JSON
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    result_record = {
        "run_id": run_id,
        "model": args.model,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "elapsed_sec": round(elapsed, 1),
        "epochs": len(history["train"]),
        "best_val_loss": round(min(history["val"]), 6),
        **metrics,
    }
    json_path = results_dir / f"{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"Hasil metrik : {json_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
