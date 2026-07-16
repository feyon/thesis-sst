from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import json
from scipy.spatial import KDTree

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_nc_path(year, raw_dir):
    anfc = raw_dir / 'cmems_analysis' / f'anfc_thetao_{year}.nc'
    if anfc.exists():
        return anfc
    glorys = raw_dir / 'cmems_reanalysis' / f'glorys_thetao_{year}.nc'
    if glorys.exists():
        return glorys
    raise FileNotFoundError(f"Tidak ada NetCDF untuk tahun {year}")

def build_kdtree(ds):
    lats = ds.latitude.values
    lons = ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return KDTree(points), lats, lons

def extract_sst(ds, tree, lats, lons, date, argo_lat, argo_lon):
    dist, idx = tree.query([argo_lat, argo_lon])
    n_lons = len(lons)
    i = idx // n_lons
    j = idx % n_lons
    try:
        val = float(
            ds['thetao'].sel(time=date.strftime('%Y-%m-%d'), method='nearest')
                        .isel(depth=0, latitude=i, longitude=j).values
        )
        return None if np.isnan(val) else round(val, 4)
    except Exception:
        return None

def metrics(obs, pred):
    bias = float(np.mean(pred - obs))
    rmse = float(np.sqrt(np.mean((pred - obs)**2)))
    mae  = float(np.mean(np.abs(pred - obs)))
    r    = float(np.corrcoef(obs, pred)[0,1]) if len(obs) > 2 else 0.0
    return dict(bias=round(bias,4), rmse=round(rmse,4),
                mae=round(mae,4), pearson_r=round(r,4), n=int(len(obs)))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    args = p.parse_args()
    cfg = load_config(args.config)

    raw_dir  = Path(cfg['data']['raw_dir'])
    proc_dir = Path(cfg['data']['processed_dir'])
    rep_dir  = Path(cfg['evaluation']['results_dir'])
    domain   = cfg['domain']
    rep_dir.mkdir(parents=True, exist_ok=True)

    test_start = pd.Timestamp(cfg['split']['test'][0])
    test_end   = pd.Timestamp(cfg['split']['test'][1])

    # Muat Argo surface (sudah bersih 1 baris per profil)
    argo = pd.read_csv(proc_dir / 'argo_float_surface.csv')
    argo['date'] = pd.to_datetime(argo['date'], errors='coerce')
    argo['lon'] = pd.to_numeric(argo['lon'], errors='coerce')
    argo['lat'] = pd.to_numeric(argo['lat'], errors='coerce')
    argo['sst_argo'] = pd.to_numeric(
        argo['sea_water_temperature_degC'], errors='coerce')
    print(f"Profil Argo total    : {len(argo):,}")

    # Filter domain & periode test
    argo = argo[
        (argo['lat'] >= domain['lat_min']) &
        (argo['lat'] <= domain['lat_max']) &
        (argo['lon'] >= domain['lon_min']) &
        (argo['lon'] <= domain['lon_max']) &
        (argo['date'] >= test_start) &
        (argo['date'] <= test_end)
    ].reset_index(drop=True)
    print(f"Dalam domain & test  : {len(argo):,} profil")
    print(f"Float unik           : {argo['bmkg_id'].nunique()}")

    # Spatial collocation per tahun
    results = []
    for year in sorted(argo['date'].dt.year.unique()):
        argo_yr = argo[argo['date'].dt.year == year]
        print(f"\nTahun {year}: {len(argo_yr)} profil...")
        try:
            nc_path = get_nc_path(year, raw_dir)
            print(f"  File: {nc_path.name}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        ds = xr.open_dataset(nc_path)
        tree, lats, lons = build_kdtree(ds)

        for _, row in argo_yr.iterrows():
            sst_cmems = extract_sst(ds, tree, lats, lons,
                                    row['date'], row['lat'], row['lon'])
            if sst_cmems is not None:
                results.append({
                    'date'      : row['date'].strftime('%Y-%m-%d'),
                    'bmkg_id'   : row['bmkg_id'],
                    'lon'       : row['lon'],
                    'lat'       : row['lat'],
                    'pressure'  : row['sea_water_pressure'],
                    'sst_argo'  : row['sst_argo'],
                    'sst_cmems' : sst_cmems,
                    'bias'      : round(sst_cmems - row['sst_argo'], 4),
                })
        ds.close()

    matched = pd.DataFrame(results)
    print(f"\nTerkolokasi          : {len(matched):,} profil")

    print("\n=== Metrik Validasi Spatial Collocation per Float ===")
    per_float = []
    for bmkg, grp in matched.groupby('bmkg_id'):
        m = metrics(grp['sst_argo'].values, grp['sst_cmems'].values)
        per_float.append({'float_id': str(bmkg), **m})
        print(f"  {str(bmkg):<20} n={m['n']:3d}  "
              f"bias={m['bias']:+.4f}  "
              f"RMSE={m['rmse']:.4f}  "
              f"r={m['pearson_r']:.4f}")

    print("\n=== Metrik Validasi Keseluruhan ===")
    overall = metrics(matched['sst_argo'].values, matched['sst_cmems'].values)
    print(f"  n={overall['n']}  "
          f"bias={overall['bias']:+.4f}C  "
          f"RMSE={overall['rmse']:.4f}C  "
          f"MAE={overall['mae']:.4f}C  "
          f"r={overall['pearson_r']:.4f}")

    matched.to_csv(rep_dir / 'argo_collocation_detail.csv', index=False)
    summary = {
        'method'     : 'spatial_collocation_kdtree',
        'overall'    : overall,
        'per_float'  : per_float,
        'test_period': [test_start.strftime('%Y-%m-%d'),
                        test_end.strftime('%Y-%m-%d')],
    }
    with open(rep_dir / 'argo_collocation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetail  : {rep_dir}/argo_collocation_detail.csv")
    print(f"Summary : {rep_dir}/argo_collocation_summary.json")

if __name__ == '__main__':
    main()
