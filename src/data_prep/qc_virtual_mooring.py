"""
QC virtual mooring: verifikasi statistik + plot visual untuk 6 titik.

Output:
  data/processed/virtual_mooring/qc/
    qc_summary_all.csv       -> describe() semua kolom, semua titik
    qc_sst_timeseries.png    -> 6 panel: sst (GLORYS/ANFC) vs sst_era5
    qc_seasonal_climatology.png -> siklus musiman rata-rata per titik
    qc_predictors.png        -> u10,v10,t2m,d2m,sp per titik (ringkas)

Jalankan dari root repo:
    python src/data_prep/qc_virtual_mooring.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
MOORING_DIR = REPO_ROOT / "data" / "processed" / "virtual_mooring"
OUT_DIR = MOORING_DIR / "qc"
OUT_DIR.mkdir(exist_ok=True)

LOKASI_MOORING = {
    "01": "Lok-1 (123.82E, 7.27S)",
    "02": "Lok-2 (125.27E, 4.97S)",
    "03": "Lok-3 (128.49E, 7.76S)",
    "04": "Lok-4 (128.75E, 5.14S)",
    "05": "Lok-5 (130.84E, 4.65S)",
    "06": "Lok-6 (130.93E, 6.13S)",
}

# ----------------------------------------------------------------------
# 1. Load semua mooring
# ----------------------------------------------------------------------
data = {}
for code, label in LOKASI_MOORING.items():
    f = MOORING_DIR / f"mooring_{code}.csv"
    df = pd.read_csv(f, index_col="date", parse_dates=True)
    data[code] = df
    print(f"{label}: {len(df)} baris, "
          f"{df.index.min().date()} s/d {df.index.max().date()}, "
          f"NaN total: {int(df.isna().sum().sum())}")

# ----------------------------------------------------------------------
# 2. Ringkasan statistik gabungan
# ----------------------------------------------------------------------
summary_rows = []
for code, df in data.items():
    desc = df.select_dtypes(include=[np.number]).describe().T
    desc.insert(0, "mooring", LOKASI_MOORING[code])
    desc.insert(1, "variabel", desc.index)
    summary_rows.append(desc.reset_index(drop=True))

summary_all = pd.concat(summary_rows, ignore_index=True)
summary_all.to_csv(OUT_DIR / "qc_summary_all.csv", index=False, float_format="%.4f")
print(f"\nRingkasan statistik -> {OUT_DIR / 'qc_summary_all.csv'}")

# Ringkasan NaN & sst_source per titik
print("\nCek NaN per kolom per titik:")
for code, df in data.items():
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    src_count = df["sst_source"].value_counts().to_dict()
    flag = "  <-- ADA NaN" if len(nan_cols) else ""
    print(f"  {LOKASI_MOORING[code]}: sst_source={src_count}{flag}")
    if len(nan_cols):
        print(f"    {nan_cols.to_dict()}")

# ----------------------------------------------------------------------
# 3. Plot 1: deret SST (GLORYS/ANFC) vs SST ERA5, 6 panel
# ----------------------------------------------------------------------
fig, axes = plt.subplots(6, 1, figsize=(13, 18), sharex=True)
for ax, (code, df) in zip(axes, data.items()):
    for source, color in (("GLORYS", "tab:blue"), ("ANFC", "tab:red")):
        sub = df[df.sst_source == source]
        ax.plot(sub.index, sub.sst, lw=0.6, color=color, label=f"SST {source}")
    ax.plot(df.index, df.sst_era5, lw=0.5, color="gray", alpha=0.6,
            label="SST ERA5 (pembanding)")
    ax.set_ylabel("SST (\u00b0C)")
    ax.set_title(LOKASI_MOORING[code], fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
axes[-1].set_xlabel("Tahun")
fig.suptitle("QC: Deret SST Virtual Mooring Laut Banda 2014-2025\n"
            "(GLORYS 2014-2023, ANFC 2024-2025, ERA5 sbg pembanding independen)",
            y=0.995)
fig.tight_layout()
fig.savefig(OUT_DIR / "qc_sst_timeseries.png", dpi=150)
plt.close(fig)
print(f"Plot deret waktu -> {OUT_DIR / 'qc_sst_timeseries.png'}")

# ----------------------------------------------------------------------
# 4. Plot 2: klimatologi musiman (rata-rata harian per hari-dalam-tahun)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 6))
for (code, df), c in zip(data.items(), colors):
    doy_mean = df.groupby(df.index.dayofyear).sst.mean()
    ax.plot(doy_mean.index, doy_mean.values, label=LOKASI_MOORING[code], color=c)
ax.set_xlabel("Hari dalam tahun")
ax.set_ylabel("SST rata-rata 2014-2025 (\u00b0C)")
ax.set_title("Klimatologi Musiman SST per Titik Mooring")
ax.axvspan(152, 273, alpha=0.08, color="blue", label="Jun-Sep (monsun tenggara)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "qc_seasonal_climatology.png", dpi=150)
plt.close(fig)
print(f"Plot klimatologi -> {OUT_DIR / 'qc_seasonal_climatology.png'}")

# ----------------------------------------------------------------------
# 5. Plot 3: ringkasan prediktor ERA5 (1 titik representatif + boxplot semua)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
vars_ = ["wind_speed", "t2m", "d2m", "sp", "u10", "v10"]
titles = ["Wind speed (m/s)", "T2m (\u00b0C)", "D2m (\u00b0C)",
          "SP (Pa)", "U10 (m/s)", "V10 (m/s)"]
for ax, var, title in zip(axes.ravel(), vars_, titles):
    box_data = [data[code][var].dropna() for code in LOKASI_MOORING]
    ax.boxplot(box_data, labels=list(LOKASI_MOORING.keys()), showfliers=False)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
fig.suptitle("Distribusi Prediktor ERA5 per Titik Mooring (2014-2025)")
fig.tight_layout()
fig.savefig(OUT_DIR / "qc_predictors.png", dpi=150)
plt.close(fig)
print(f"Plot prediktor -> {OUT_DIR / 'qc_predictors.png'}")

print(f"\nSelesai. Semua output QC di: {OUT_DIR}/")
print("\nChecklist manual yang perlu dicek dari plot:")
print("  1. qc_sst_timeseries: SST GLORYS(biru) & ERA5(abu2) saling dekat?")
print("     Sambungan GLORYS->ANFC (biru->merah) di awal 2024 mulus?")
print("  2. qc_seasonal_climatology: Lok-3 & Lok-6 (zona upwelling) turun")
print("     lebih dalam saat Jun-Sep dibanding titik lain?")
print("  3. qc_predictors: tidak ada outlier ekstrem yang mencurigakan")
print("     (spike jauh dari boxplot pada wind_speed/sp)?")