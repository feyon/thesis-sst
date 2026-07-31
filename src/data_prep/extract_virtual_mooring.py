"""
Ekstraksi deret waktu virtual mooring di 6 titik Laut Banda.

Sumber data:
  1. GLORYS reanalysis  : glorys_thetao_2014..2023.nc  -> SST target (train/val/test)
  2. CMEMS ANFC         : anfc_thetao_2024..2025.nc    -> SST uji operasional
                          (dibatasi s/d 2025-12-31; file anfc_thetao_2026.nc
                          sengaja tidak dipakai dalam project ini)
  3. ERA5 6-hourly      : era5_banda_2014..2025.nc     -> prediktor instant
                          (u10, v10, t2m, d2m, sp; sst ERA5 disimpan sbg pembanding)

Output: data/processed/virtual_mooring/mooring_XX.csv
Kolom : date, sst (GLORYS/ANFC), sst_source, sst_era5, u10, v10, wind_speed,
        t2m, d2m, sp

Kebijakan penting:
- Nearest valid ocean cell: jika grid cell terdekat di suatu produk NaN
  (land mask), dicari cell laut valid terdekat dalam radius pencarian.
  Koordinat grid aktual yang terpakai dicatat ke mooring_grid_info.csv
  (lampiran thesis).
- Agregasi harian ERA5 instant: mean dari 4 sampel (00,06,12,18 UTC).
- Variabel radiasi (ssr, str) TIDAK disertakan karena kendala akses data
  daily-aggregated. Sinyal musiman radiasi diasumsikan tertangkap secara
  implisit melalui fitur temporal (encoding hari-dalam-tahun, ditambahkan
  di tahap windowing/feature engineering) dan variabel suhu udara/
  kelembapan (t2m, d2m) yang berkorelasi dengan heat flux permukaan.
- GLORYS & ANFC TIDAK digabung mulus: kolom sst_source menandai asal data
  agar analisis distribution shift tetap bisa dilakukan.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# ----------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------
# Root repo diturunkan otomatis dari lokasi file ini:
# src/data_prep/extract_virtual_mooring.py -> naik 2 level -> ~/thesis-sst
REPO_ROOT = Path(__file__).resolve().parents[2]

BASE = REPO_ROOT / "data" / "raw"
DIR_GLORYS = BASE / "cmems_reanalysis"
DIR_ANFC = BASE / "cmems_analysis"
DIR_ERA5 = BASE / "dataset_laut_banda" / "era5_banda_fixed"
OUT_DIR = REPO_ROOT / "data" / "processed" / "virtual_mooring"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOKASI_MOORING = {
    "Lok-1": {"lon": 123.82, "lat": -7.27},
    "Lok-2": {"lon": 125.27, "lat": -4.97},
    "Lok-3": {"lon": 128.49, "lat": -7.76},
    "Lok-4": {"lon": 128.75, "lat": -5.14},
    "Lok-5": {"lon": 130.84, "lat": -4.65},
    "Lok-6": {"lon": 130.93, "lat": -6.13},
}

MAX_SEARCH_CELLS = 3  # radius pencarian nearest ocean cell (dlm jumlah grid)


# ----------------------------------------------------------------------
# Util
# ----------------------------------------------------------------------
def open_multi(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Tidak ada file cocok: {pattern}")
    print(f"  membuka {len(files)} file: {os.path.basename(files[0])} .. "
          f"{os.path.basename(files[-1])}")
    return xr.open_mfdataset(files, combine="by_coords")


def coord_names(ds):
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    time = "time" if "time" in ds.coords else "valid_time"
    return lat, lon, time


def nearest_ocean_cell(da, lat0, lon0, latn, lonn):
    """Cari grid cell valid (bukan NaN) terdekat dari (lat0, lon0).

    Mengembalikan (lat_grid, lon_grid) yang benar-benar terpakai.
    Validitas dicek pada timestep pertama.
    """
    sample = da.isel({[d for d in da.dims if d not in (latn, lonn)][0]: 0}) \
        if any(d not in (latn, lonn) for d in da.dims) else da
    sample = sample.load()

    lats = da[latn].values
    lons = da[lonn].values
    i0 = int(np.abs(lats - lat0).argmin())
    j0 = int(np.abs(lons - lon0).argmin())

    for r in range(0, MAX_SEARCH_CELLS + 1):
        best = None
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                i, j = i0 + di, j0 + dj
                if not (0 <= i < len(lats) and 0 <= j < len(lons)):
                    continue
                val = sample.isel({latn: i, lonn: j}).values
                if np.isfinite(val).all() if np.ndim(val) else np.isfinite(val):
                    d = (lats[i] - lat0) ** 2 + (lons[j] - lon0) ** 2
                    if best is None or d < best[0]:
                        best = (d, float(lats[i]), float(lons[j]))
        if best is not None:
            return best[1], best[2]
    raise ValueError(f"Tidak ada ocean cell valid dlm radius "
                     f"{MAX_SEARCH_CELLS} grid dari ({lat0}, {lon0})")


def extract_point_series(da, lat_g, lon_g, latn, lonn, timen):
    s = da.sel({latn: lat_g, lonn: lon_g}, method="nearest").load()
    idx = pd.to_datetime(s[timen].values)
    return pd.Series(np.asarray(s.values).ravel(), index=idx)


# ----------------------------------------------------------------------
# 1. SST: GLORYS (2014-2023) + ANFC (2024-2025)
# ----------------------------------------------------------------------
print("== Membuka dataset ==")
print("GLORYS:")
ds_glorys = open_multi(os.path.join(DIR_GLORYS, "glorys_thetao_*.nc"))
print("ANFC:")
ds_anfc = open_multi(os.path.join(DIR_ANFC, "anfc_thetao_*.nc"))
print("ERA5 6-hourly (instant vars):")
ds_era5 = open_multi(
    os.path.join(DIR_ERA5, "era5_banda_*_parts",
                 "data_stream-oper_stepType-instant.nc")
)

# ERA5T (data ~3 bulan terakhir, biasanya thn 2024-2025) kadang punya
# dimensi tambahan 'expver' (1=ERA5 final, 5=ERA5T preliminer) yang
# membuat setiap variabel pecah ganda dengan NaN saling melengkapi.
# Digabung di sini agar deret waktu tetap tunggal per titik.
if "expver" in ds_era5.dims:
    print("  -> dimensi 'expver' terdeteksi (ERA5T), digabung otomatis")
    parts = [ds_era5.sel(expver=v).dropna(dim="time", how="all")
             for v in ds_era5.expver.values]
    ds_era5 = xr.concat(parts, dim="time").sortby("time")
    ds_era5 = ds_era5.drop_duplicates(dim="time")


def pick_sst_var(ds):
    for v in ("thetao", "analysed_sst", "sst"):
        if v in ds:
            return v
    raise KeyError(f"Variabel SST tidak ditemukan. Tersedia: {list(ds.data_vars)}")


def squeeze_depth(da):
    for d in ("depth", "lev"):
        if d in da.dims:
            da = da.isel({d: 0})  # level permukaan
    return da


sst_glorys = squeeze_depth(ds_glorys[pick_sst_var(ds_glorys)])
sst_anfc = squeeze_depth(ds_anfc[pick_sst_var(ds_anfc)])

# ----------------------------------------------------------------------
# 2. Loop 6 titik
# ----------------------------------------------------------------------
grid_info = []

for name, p in LOKASI_MOORING.items():
    print(f"\n== {name} ({p['lat']}, {p['lon']}) ==")
    lat0, lon0 = p["lat"], p["lon"]

    # --- GLORYS ---
    latn, lonn, timen = coord_names(ds_glorys)
    g_lat, g_lon = nearest_ocean_cell(sst_glorys, lat0, lon0, latn, lonn)
    s_glorys = extract_point_series(sst_glorys, g_lat, g_lon, latn, lonn, timen)
    grid_info.append([name, "GLORYS", lat0, lon0, g_lat, g_lon])
    print(f"  GLORYS grid: ({g_lat:.4f}, {g_lon:.4f})")

    # --- ANFC ---
    latn, lonn, timen = coord_names(ds_anfc)
    a_lat, a_lon = nearest_ocean_cell(sst_anfc, lat0, lon0, latn, lonn)
    s_anfc = extract_point_series(sst_anfc, a_lat, a_lon, latn, lonn, timen)
    grid_info.append([name, "ANFC", lat0, lon0, a_lat, a_lon])
    print(f"  ANFC grid  : ({a_lat:.4f}, {a_lon:.4f})")

    # Gabung SST + penanda sumber (ANFC hanya dipakai setelah GLORYS habis)
    # Batas atas ANFC: hanya sampai 2025-12-31 (anfc_thetao_2026.nc
    # sengaja tidak dipakai dalam project ini)
    ANFC_END = pd.Timestamp("2025-12-31")
    cutoff = s_glorys.index.max()
    s_anfc_use = s_anfc[(s_anfc.index > cutoff) & (s_anfc.index <= ANFC_END)]
    sst = pd.concat([s_glorys, s_anfc_use]).sort_index()
    sst_source = pd.Series(
        np.where(sst.index <= cutoff, "GLORYS", "ANFC"), index=sst.index
    )
    # Normalisasi ke tanggal (daily)
    sst.index = sst.index.normalize()
    sst_source.index = sst_source.index.normalize()

    # --- ERA5 instant (6-hourly -> daily mean) ---
    latn, lonn, timen = coord_names(ds_era5)
    e_lat, e_lon = nearest_ocean_cell(ds_era5["sst"], lat0, lon0, latn, lonn)
    grid_info.append([name, "ERA5", lat0, lon0, e_lat, e_lon])
    print(f"  ERA5 grid  : ({e_lat:.4f}, {e_lon:.4f})")

    era5_daily = {}
    for var in ("u10", "v10", "t2m", "d2m", "sp", "sst"):
        s = extract_point_series(ds_era5[var], e_lat, e_lon, latn, lonn, timen)
        era5_daily[var] = s.resample("1D").mean()
    df_era5 = pd.DataFrame(era5_daily)
    df_era5["wind_speed"] = np.sqrt(df_era5.u10**2 + df_era5.v10**2)
    # Kelvin -> Celsius utk suhu
    for v in ("t2m", "d2m", "sst"):
        df_era5[v] = df_era5[v] - 273.15
    df_era5 = df_era5.rename(columns={"sst": "sst_era5"})

    # --- Merge final ---
    df = pd.DataFrame({"sst": sst, "sst_source": sst_source})
    df = df.join(df_era5, how="left")
    df.index.name = "date"

    out = os.path.join(OUT_DIR, f"mooring_{name.replace('Lok-', '0')}.csv")
    df.to_csv(out, float_format="%.4f")
    n_nan = int(df["sst"].isna().sum())
    print(f"  -> {out}  ({len(df)} hari, {df.index.min().date()} s/d "
          f"{df.index.max().date()}, NaN sst: {n_nan})")

# ----------------------------------------------------------------------
# 3. Simpan info grid aktual (lampiran thesis)
# ----------------------------------------------------------------------
pd.DataFrame(
    grid_info,
    columns=["mooring", "produk", "lat_target", "lon_target",
             "lat_grid", "lon_grid"],
).to_csv(os.path.join(OUT_DIR, "mooring_grid_info.csv"), index=False)

print(f"\nSelesai. Output di: {OUT_DIR}/")
print("File mooring_grid_info.csv berisi koordinat grid aktual per produk.")