"""Unduh data CMEMS untuk domain Laut Banda (pendekatan univariat: hanya thetao).

Strategi: loop per tahun (menghindari satu file raksasa & memudahkan resume).
  - Reanalysis GLORYS12V1  : 2014 .. reanalysis_end
  - Analysis  ANFC         : 2024 .. end_date
Jalankan:  python -m src.data.download --config configs/config.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _year_range(start: str, end: str):
    y0, y1 = int(start[:4]), int(end[:4])
    return range(y0, y1 + 1)


def download(cfg: dict) -> None:
    import copernicusmarine as cm  # impor di dalam agar config bisa dibaca tanpa lib

    d, data = cfg["domain"], cfg["data"]
    raw = Path(data["raw_dir"]); raw.mkdir(parents=True, exist_ok=True)

    box = dict(
        variables=data["variables"],
        minimum_longitude=d["buffer_lon_min"],
        maximum_longitude=d["buffer_lon_max"],
        minimum_latitude=d["buffer_lat_min"],
        maximum_latitude=d["buffer_lat_max"],
        minimum_depth=d["depth_min"],
        maximum_depth=d["depth_max"],
        output_directory=str(raw),
    )

    rean_end_year = int(data["reanalysis_end"][:4])

    # 1) Reanalysis
    for year in _year_range(data["start_date"], data["reanalysis_end"]):
        fname = f"glorys_thetao_{year}.nc"
        if (raw / fname).exists():
            print(f"  [skip] {fname} sudah ada"); continue
        print(f"[reanalysis] mengunduh {year} ...")
        cm.subset(
            dataset_id=data["reanalysis_dataset_id"],
            start_datetime=f"{year}-01-01T00:00:00",
            end_datetime=f"{year}-12-31T23:59:59",
            output_filename=fname,
            **box,
        )

    # 2) Analysis (ANFC) — mulai tahun setelah reanalysis_end
    for year in _year_range(f"{rean_end_year + 1}-01-01", data["end_date"]):
        fname = f"anfc_thetao_{year}.nc"
        if (raw / fname).exists():
            print(f"  [skip] {fname} sudah ada"); continue
        print(f"[analysis]   mengunduh {year} ...")
        cm.subset(
            dataset_id=data["analysis_dataset_id"],
            start_datetime=f"{year}-01-01T00:00:00",
            end_datetime=f"{year}-12-31T23:59:59",
            output_filename=fname,
            **box,
        )

    print("Selesai. File mentah tersimpan di:", raw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    download(load_config(p.parse_args().config))
