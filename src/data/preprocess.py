"""Pra-pemrosesan SST univariat: NetCDF mentah -> deret waktu SST harian bersih.

Tahapan:
  1. Muat semua file NetCDF (reanalysis + analysis) via xarray.
  2. Pilih variabel thetao pada lapisan permukaan, potong ke domain analisis.
  3. Quality control: nilai di luar [sst_min, sst_max] -> NaN.
  4. Reduksi spasial -> rata-rata domain (deret waktu tunggal, representatif).
     (Catatan: untuk versi per-titik/Argo, ganti reduksi ini dengan ekstraksi
      pada titik grid terpilih -> array [time, n_points]. Lihat README.)
  5. Interpolasi temporal celah kecil + simpan ke data/processed/sst_series.csv

Jalankan:  python -m src.data.preprocess --config configs/config.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_series(cfg: dict) -> pd.DataFrame:
    import xarray as xr

    d, data, qc = cfg["domain"], cfg["data"], cfg["qc"]
    raw = Path(data["raw_dir"])
    files = sorted(raw.glob("*.nc"))
    if not files:
        raise FileNotFoundError(
            f"Tidak ada file .nc di {raw}. Jalankan download.py lebih dulu."
        )

    print(f"Memuat {len(files)} file NetCDF ...")
    ds = xr.open_mfdataset(files, combine="by_coords")

    # Pilih lapisan permukaan jika ada dimensi kedalaman
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)

    # Potong ke domain analisis (bukan buffer)
    ds = ds.sel(
        latitude=slice(d["lat_min"], d["lat_max"]),
        longitude=slice(d["lon_min"], d["lon_max"]),
    )

    sst = ds["thetao"]

    # 3. Quality control
    sst = sst.where((sst >= qc["sst_min"]) & (sst <= qc["sst_max"]))

    # 4. Reduksi spasial -> rata-rata domain harian
    series = sst.mean(dim=["latitude", "longitude"], skipna=True).to_series()
    series.index = pd.to_datetime(series.index).normalize()
    series = series[~series.index.duplicated(keep="first")].sort_index()

    # Reindex ke grid harian penuh
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_idx)

    # 5. Interpolasi temporal celah kecil
    n_missing = int(series.isna().sum())
    if qc.get("interpolate_gaps", True) and n_missing:
        series = series.interpolate(method="time", limit_direction="both")
    print(f"Deret waktu SST: {len(series)} hari, {n_missing} nilai diinterpolasi.")

    return pd.DataFrame({"date": series.index, "sst": series.values})


def main(cfg: dict) -> None:
    df = build_series(cfg)
    out = Path(cfg["data"]["processed_dir"]); out.mkdir(parents=True, exist_ok=True)
    fpath = out / "sst_series.csv"
    df.to_csv(fpath, index=False)
    print("Tersimpan:", fpath)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    main(load_config(p.parse_args().config))
