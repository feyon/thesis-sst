from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import json

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def load_argo_surface(path):
    df = pd.read_csv(path)
    df['MESSAGEDATE'] = pd.to_datetime(df['MESSAGEDATE'], errors='coerce')
    df['date'] = df['MESSAGEDATE'].dt.date.astype(str)
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['pressure'] = pd.to_numeric(df['sea_water_pressure'], errors='coerce')
    df['sst_argo'] = pd.to_numeric(df['sea_water_temperature_degC'], errors='coerce')
    profile_keys = ['date', 'imei', 'lon', 'lat']
    surface = (
        df.dropna(subset=['pressure', 'sst_argo'])
          .sort_values('pressure')
          .groupby(profile_keys, as_index=False)
          .first()
    )[['date', 'imei', 'bmkg_id', 'lon', 'lat', 'pressure', 'sst_argo']]
    return surface

def metrics(obs, pred):
    bias = float(np.mean(pred - obs))
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    mae  = float(np.mean(np.abs(pred - obs)))
    r    = float(np.corrcoef(obs, pred)[0, 1])
    return dict(bias=round(bias,4), rmse=round(rmse,4),
                mae=round(mae,4), pearson_r=round(r,4), n=int(len(obs)))

def validate(cfg):
    proc_dir = Path(cfg['data']['processed_dir'])
    rep_dir  = Path(cfg['evaluation']['results_dir'])
    rep_dir.mkdir(parents=True, exist_ok=True)
    domain     = cfg['domain']
    test_start = cfg['split']['test'][0]
    test_end   = cfg['split']['test'][1]

    argo = load_argo_surface(proc_dir / 'argo_float_surface.csv')
    print(f"Profil Argo total    : {len(argo):,}")

    argo = argo[
        (argo['lat'] >= domain['lat_min']) &
        (argo['lat'] <= domain['lat_max']) &
        (argo['lon'] >= domain['lon_min']) &
        (argo['lon'] <= domain['lon_max']) &
        (argo['date'] >= test_start) &
        (argo['date'] <= test_end)
    ].reset_index(drop=True)
    print(f"Dalam domain & test  : {len(argo):,} profil")
    print(f"Float unik           : {argo['imei'].nunique()}")

    sst = pd.read_csv(proc_dir / 'sst_series.csv')
    sst['date'] = sst['date'].astype(str)
    sst_dict = dict(zip(sst['date'], sst['sst']))

    argo['sst_model'] = argo['date'].map(sst_dict)
    matched = argo.dropna(subset=['sst_model']).copy()
    print(f"Terkolokasi          : {len(matched):,} profil")

    print("\n=== Metrik Validasi per Float ===")
    per_float = []
    for imei, grp in matched.groupby('imei'):
        bmkg = grp['bmkg_id'].iloc[0]
        m = metrics(grp['sst_argo'].values, grp['sst_model'].values)
        per_float.append({'float_id': bmkg, **m})
        print(f"  {bmkg:<20} n={m['n']:3d}  bias={m['bias']:+.4f}  RMSE={m['rmse']:.4f}  r={m['pearson_r']:.4f}")

    print("\n=== Metrik Validasi Keseluruhan ===")
    overall = metrics(matched['sst_argo'].values, matched['sst_model'].values)
    print(f"  n={overall['n']}  bias={overall['bias']:+.4f}C  RMSE={overall['rmse']:.4f}C  MAE={overall['mae']:.4f}C  r={overall['pearson_r']:.4f}")

    matched.to_csv(rep_dir / 'argo_validation_detail.csv', index=False)
    summary = {'overall': overall, 'per_float': per_float,
                'test_period': [test_start, test_end]}
    with open(rep_dir / 'argo_validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetail  : {rep_dir}/argo_validation_detail.csv")
    print(f"Summary : {rep_dir}/argo_validation_summary.json")
    return matched

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    args = p.parse_args()
    validate(load_config(args.config))
