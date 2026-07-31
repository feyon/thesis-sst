"""
Bangun dataset sliding-window dari virtual mooring, mengikuti config.yaml
resmi project (single source of truth).

Menghasilkan seluruh kombinasi grid eksperimen:
    len(experiment.lookback_windows) x len(experiment.horizons)
misal 4 x 4 = 16 kombinasi (lookback, horizon), masing2 dgn 3 split
(train/val/test) sesuai section `split:` di config.

Fitur mengikuti project.approach:
    - univariate   -> hanya kolom data.target_col (F=1, TANPA fitur temporal,
                       persis sesuai data.variables: [thetao] di config)
    - multivariate -> data.target_col + data.predictor_vars (+ doy_sin/cos
                       kalau data.add_temporal_features_if_multivariate: true)

QC (section `qc:`) diterapkan ke SST SEBELUM windowing:
    - nilai di luar [sst_min, sst_max] -> NaN
    - gap kecil (<= qc.max_gap_days) -> interpolasi linear
    - gap besar dibiarkan NaN, baris terkait didrop saat windowing

Cara jalan (lokal):
    python -m src.data_prep.build_windows --config configs/config.yaml

Cara jalan (SLURM + apptainer):
    cd ~/thesis-sst
    srun --cpus-per-task=4 --mem=8G --time=00:20:00 \\
      apptainer exec \\
      --bind ~/thesis-sst/data:/workspace/data \\
      --bind ~/thesis-sst/configs:/workspace/configs \\
      --bind ~/thesis-sst/src:/workspace/src \\
      --bind ~/thesis-sst/results:/workspace/results \\
      ~/thesis-sst/thesis-sst.sif \\
      python -m src.data_prep.build_windows --config configs/config.yaml

Output: <data.windowed_dir>/lb{N:02d}_h{H:02d}/
    X_train.npy, y_train.npy, meta_train.csv
    X_val.npy,   y_val.npy,   meta_val.csv
    X_test.npy,  y_test.npy,  meta_test.csv
Plus satu file scaler bersama (fitur sama di semua kombinasi lookback/horizon):
    <data.windowed_dir>/scaler_params.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Config dibaca dari: {config_path}")
    print(f"  project.approach = {cfg['project']['approach']}")
    return cfg


def apply_qc(df, qc_cfg, target_col):
    """Terapkan batas fisik + interpolasi gap kecil pada kolom target."""
    sst_min = qc_cfg.get("sst_min", -999)
    sst_max = qc_cfg.get("sst_max", 999)
    max_gap = qc_cfg.get("max_gap_days", 3)
    interpolate = qc_cfg.get("interpolate_gaps", True)

    df = df.copy()
    out_of_range = (df[target_col] < sst_min) | (df[target_col] > sst_max)
    n_bad = int(out_of_range.sum())
    if n_bad:
        print(f"    QC: {n_bad} nilai {target_col} di luar [{sst_min},{sst_max}] -> NaN")
        df.loc[out_of_range, target_col] = np.nan

    if interpolate and df[target_col].isna().any():
        df[target_col] = df[target_col].interpolate(
            method="linear", limit=max_gap, limit_direction="both"
        )
    return df


def add_temporal_features(df):
    doy = df.index.dayofyear
    df = df.copy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def make_windows(arr_feat, arr_target, n_in, n_out):
    X, y = [], []
    T = len(arr_feat)
    for t in range(T - n_in - n_out + 1):
        X.append(arr_feat[t: t + n_in])
        y.append(arr_target[t + n_in: t + n_in + n_out])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=None,
                        help="Override experiment.lookback_windows (ad-hoc, "
                             "TIDAK mengubah config.yaml). Mis. --lookbacks 1 "
                             "utk tambah satu nilai tanpa ganggu grid utama.")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Override experiment.horizons (ad-hoc).")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    qc_cfg = cfg.get("qc", {})
    split_cfg = {k: tuple(v) for k, v in cfg["split"].items()}
    exp_cfg = cfg["experiment"]
    approach = cfg["project"]["approach"]

    mooring_dir = REPO_ROOT / data_cfg["mooring_dir"]
    windowed_dir = REPO_ROOT / data_cfg["windowed_dir"]
    windowed_dir.mkdir(parents=True, exist_ok=True)

    mooring_codes = list(data_cfg["mooring_codes"])
    target_col = data_cfg["target_col"]

    if approach == "multivariate":
        predictor_vars = list(data_cfg.get("predictor_vars", []))
        use_temporal = data_cfg.get("add_temporal_features_if_multivariate", True)
    else:
        predictor_vars = []
        use_temporal = False

    feat_cols_final = [target_col] + predictor_vars
    if use_temporal:
        feat_cols_final += ["doy_sin", "doy_cos"]

    print(f"  Feature set ({len(feat_cols_final)}, F={len(feat_cols_final)}): "
          f"{feat_cols_final}")

    # ------------------------------------------------------------------
    print("\nMemuat & QC data mooring...")
    all_df = {}
    for code in mooring_codes:
        f = mooring_dir / f"mooring_{code}.csv"
        df = pd.read_csv(f, index_col="date", parse_dates=True)
        print(f"  mooring_{code}: {len(df)} baris")
        df = apply_qc(df, qc_cfg, target_col)
        if use_temporal:
            df = add_temporal_features(df)
        all_df[code] = df

    # ------------------------------------------------------------------
    print("\nFitting normalisasi (z-score) dari periode train saja "
          "(berlaku sama utk semua kombinasi lookback/horizon)...")
    train_start, train_end = split_cfg["train"]
    train_concat = pd.concat([
        df.loc[train_start:train_end, feat_cols_final] for df in all_df.values()
    ])
    scaler_mean = train_concat.mean()
    scaler_std = train_concat.std()
    pd.DataFrame({"mean": scaler_mean, "std": scaler_std}).to_csv(
        windowed_dir / "scaler_params.csv"
    )
    print(f"  Parameter scaler -> {windowed_dir / 'scaler_params.csv'}")

    def normalize(df):
        return (df[feat_cols_final] - scaler_mean) / scaler_std

    # ------------------------------------------------------------------
    lookback_windows = args.lookbacks or exp_cfg["lookback_windows"]
    horizons = args.horizons or exp_cfg["horizons"]
    if args.lookbacks:
        print(f"  [override CLI] lookback_windows = {lookback_windows} "
             f"(config asli: {exp_cfg['lookback_windows']})")
    if args.horizons:
        print(f"  [override CLI] horizons = {horizons} "
             f"(config asli: {exp_cfg['horizons']})")
    total_combo = len(lookback_windows) * len(horizons)
    combo_i = 0

    for n_in in lookback_windows:
        for n_out in horizons:
            combo_i += 1
            combo_dir = windowed_dir / f"lb{n_in:02d}_h{n_out:02d}"
            combo_dir.mkdir(exist_ok=True)
            print(f"\n=== Kombinasi {combo_i}/{total_combo}: "
                  f"lookback={n_in} hari, horizon={n_out} hari ===")

            for split_name, (start, end) in split_cfg.items():
                start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
                X_list, y_list, meta_rows = [], [], []

                for code, df in all_df.items():
                    ctx_start = start_ts - pd.Timedelta(days=n_in)
                    sub = df.loc[ctx_start:end_ts]
                    if sub[feat_cols_final].isna().any().any():
                        sub = sub.dropna(subset=feat_cols_final)
                    if len(sub) < n_in + n_out:
                        continue

                    norm = normalize(sub)
                    target_raw = sub[target_col].values
                    Xw, yw = make_windows(norm.values, target_raw, n_in, n_out)

                    tgt_dates = sub.index[n_in: n_in + len(yw)]
                    mask = (tgt_dates >= start_ts) & (tgt_dates <= end_ts)
                    Xw, yw = Xw[mask], yw[mask]
                    kept_dates = tgt_dates[mask]

                    mooring_id = np.full(len(Xw), mooring_codes.index(code))
                    X_list.append(Xw)
                    y_list.append(yw)
                    meta_rows.append(pd.DataFrame({
                        "mooring_id": mooring_id,
                        "mooring_code": code,
                        "target_start_date": kept_dates,
                    }))

                if not X_list:
                    print(f"  split '{split_name}': kosong, dilewati.")
                    continue

                X_all = np.concatenate(X_list, axis=0)
                y_all = np.concatenate(y_list, axis=0)
                meta_all = pd.concat(meta_rows, ignore_index=True)

                np.save(combo_dir / f"X_{split_name}.npy", X_all)
                np.save(combo_dir / f"y_{split_name}.npy", y_all)
                meta_all.to_csv(combo_dir / f"meta_{split_name}.csv", index=False)
                print(f"  {split_name}: X{X_all.shape} y{y_all.shape} "
                      f"-> {combo_dir.name}/")

    print(f"\nSelesai. {total_combo} kombinasi lookback x horizon "
          f"dihasilkan di: {windowed_dir}/")
    print(f"Approach: {approach}, F={len(feat_cols_final)} fitur: {feat_cols_final}")
    print("Target TIDAK dinormalisasi (skala Celsius asli).")


if __name__ == "__main__":
    main()