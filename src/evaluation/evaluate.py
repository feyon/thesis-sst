"""
Evaluasi satu checkpoint model terlatih terhadap baseline Persistence,
menghitung seluruh metrik di config.evaluation.metrics
(rmse, mae, r2, nse, skill_score), dipecah per horizon-step dan per mooring.

Cara jalan (lokal / CPU):
    python -m src.evaluation.evaluate --config configs/config.yaml \\
        --lookback 30 --horizon 7 --model hybrid --split test

Cara jalan (SLURM + apptainer, GPU):
    cd ~/thesis-sst
    srun --partition=gpu_riset --gres=gpu:1 --cpus-per-task=4 --mem=16G \\
      --time=00:30:00 \\
      apptainer exec --nv --pwd /workspace \\
      --bind ~/thesis-sst/data:/workspace/data \\
      --bind ~/thesis-sst/configs:/workspace/configs \\
      --bind ~/thesis-sst/src:/workspace/src \\
      --bind ~/thesis-sst/results:/workspace/results \\
      ~/thesis-sst/thesis-sst.sif \\
      python -m src.evaluation.evaluate --config configs/config.yaml \\
        --lookback 30 --horizon 7 --model hybrid --split test

Output: <results_dir>/evaluation/<model>_lb{N}_h{H}_<split>/
    metrics_overall.csv     -> 1 baris: model vs persistence, semua metrik
    metrics_per_horizon.csv -> metrik per langkah horizon (h=1..N_OUT)
    metrics_per_mooring.csv -> metrik per titik mooring (Lok-1..6)
    predictions.npz         -> y_true, y_pred, y_persistence, mooring_id,
                                target_start_date (utk analisis lanjutan)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training.train import (  # noqa: E402
    WindowDataset, build_model, load_split, resolve_device,
)


# ==========================================================================
# Metrik
# ==========================================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%). Aman dipakai di sini karena
    SST dlm Celsius selalu jauh dari nol (~25-32 C di Laut Banda),
    tidak ada risiko pembagian dgn nilai mendekati nol."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def r2_pearson(y_true, y_pred):
    """R^2 = korelasi Pearson kuadrat (kekuatan hubungan linear)."""
    if np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
        return float("nan")
    corr = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]
    return float(corr ** 2)


def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency: 1 - SS_res/SS_tot (menghukum bias)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-9:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def skill_score(y_true, y_pred, y_ref):
    """Murphy Skill Score relatif thd baseline (Persistence).
    SS = 1 - MSE_model / MSE_reference. SS=0 -> setara Persistence,
    SS>0 -> lebih baik, SS<0 -> lebih buruk dari Persistence."""
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_ref = np.mean((y_true - y_ref) ** 2)
    if mse_ref < 1e-9:
        return float("nan")
    return float(1 - mse_model / mse_ref)


def compute_all_metrics(y_true, y_pred, y_persistence):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_pearson(y_true, y_pred),
        "nse": nse(y_true, y_pred),
        "skill_score": skill_score(y_true, y_pred, y_persistence),
        "rmse_persistence": rmse(y_true, y_persistence),
        "mae_persistence": mae(y_true, y_persistence),
        "mape_persistence": mape(y_true, y_persistence),
    }


# ==========================================================================
# Inference
# ==========================================================================
def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            preds.append(model(X).cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["lstm", "transformer", "hybrid"])
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    device = resolve_device(train_cfg["device"])
    print(f"Device dipakai: {device}")

    windowed_dir = REPO_ROOT / cfg["data"]["windowed_dir"]
    results_dir = REPO_ROOT / cfg["evaluation"]["results_dir"]
    target_col = cfg["data"]["target_col"]
    threshold = cfg["evaluation"]["skill_score_threshold"]

    run_name = f"{args.model}_lb{args.lookback:02d}_h{args.horizon:02d}"
    ckpt_path = results_dir / "checkpoints" / run_name / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {ckpt_path}\n"
            f"Jalankan training dulu utk kombinasi ini."
        )

    out_dir = results_dir / "evaluation" / f"{run_name}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print(f"\n=== Evaluasi: {run_name}, split='{args.split}' ===")
    X, y = load_split(windowed_dir, args.lookback, args.horizon, args.split)
    meta = pd.read_csv(
        windowed_dir / f"lb{args.lookback:02d}_h{args.horizon:02d}" /
        f"meta_{args.split}.csv"
    )
    print(f"  X {X.shape}, y {y.shape}, meta {len(meta)} baris")
    assert len(meta) == len(X), "Jumlah baris meta tidak cocok dgn X/y!"

    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config_model"], ckpt["input_size"],
                        y.shape[-1], cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Checkpoint dimuat: epoch {ckpt['epoch']}, "
          f"val_rmse (saat training) {ckpt['val_rmse']:.4f}")

    loader = DataLoader(WindowDataset(X, y), batch_size=train_cfg["batch_size"],
                        shuffle=False, num_workers=0)

    y_pred = predict(model, loader, device)  # (N, n_out)
    y_true = y                                # (N, n_out), skala asli

    # ------------------------------------------------------------------
    # Baseline Persistence: SST hari terakhir di window input (kolom
    # target_col SELALU index 0 pada feature set, sesuai build_windows.py)
    scaler = pd.read_csv(windowed_dir / "scaler_params.csv", index_col=0)
    sst_mean = scaler.loc[target_col, "mean"]
    sst_std = scaler.loc[target_col, "std"]

    sst_idx = 0  # target_col selalu fitur pertama di build_windows.py
    last_sst_norm = X[:, -1, sst_idx]                 # (N,)
    last_sst_raw = last_sst_norm * sst_std + sst_mean  # un-normalize -> °C
    y_persistence = np.repeat(last_sst_raw[:, None], y.shape[-1], axis=1)

    # ------------------------------------------------------------------
    print("\nMenghitung metrik...")
    overall = compute_all_metrics(y_true, y_pred, y_persistence)
    overall_row = {"run_name": run_name, "split": args.split, **overall}
    pd.DataFrame([overall_row]).to_csv(out_dir / "metrics_overall.csv", index=False)

    print(f"  RMSE model={overall['rmse']:.4f}  RMSE persistence="
          f"{overall['rmse_persistence']:.4f}")
    print(f"  MAE  model={overall['mae']:.4f}   MAE  persistence="
          f"{overall['mae_persistence']:.4f}")
    print(f"  MAPE model={overall['mape']:.2f}%  MAPE persistence="
          f"{overall['mape_persistence']:.2f}%")
    print(f"  R2={overall['r2']:.4f}  NSE={overall['nse']:.4f}  "
          f"Skill Score={overall['skill_score']:.4f}")

    verdict = "LOLOS" if overall["skill_score"] >= threshold else "BELUM LOLOS"
    print(f"  Gate skill_score_threshold={threshold} -> {verdict}")

    # ------------------------------------------------------------------
    per_h_rows = []
    for h in range(y.shape[-1]):
        m = compute_all_metrics(y_true[:, h], y_pred[:, h], y_persistence[:, h])
        per_h_rows.append({"horizon_step": h + 1, **m})
    pd.DataFrame(per_h_rows).to_csv(out_dir / "metrics_per_horizon.csv", index=False)
    print(f"\n  Metrik per horizon-step -> {out_dir / 'metrics_per_horizon.csv'}")

    # ------------------------------------------------------------------
    per_mooring_rows = []
    for mid in sorted(meta["mooring_id"].unique()):
        mask = (meta["mooring_id"] == mid).values
        code = f"{int(mid) + 1:02d}"  # derive langsung, hindari isu leading-zero
                                       # yang hilang saat pandas parse CSV
        m = compute_all_metrics(y_true[mask], y_pred[mask], y_persistence[mask])
        per_mooring_rows.append({"mooring_id": mid, "mooring_code": code, **m})
    pd.DataFrame(per_mooring_rows).to_csv(
        out_dir / "metrics_per_mooring.csv", index=False
    )
    print(f"  Metrik per mooring     -> {out_dir / 'metrics_per_mooring.csv'}")

    # ------------------------------------------------------------------
    np.savez(
        out_dir / "predictions.npz",
        y_true=y_true, y_pred=y_pred, y_persistence=y_persistence,
        mooring_id=meta["mooring_id"].values,
        target_start_date=meta["target_start_date"].values,
    )
    print(f"  Prediksi mentah        -> {out_dir / 'predictions.npz'}")

    print(f"\nSelesai. Semua output evaluasi di: {out_dir}/")


if __name__ == "__main__":
    main()
