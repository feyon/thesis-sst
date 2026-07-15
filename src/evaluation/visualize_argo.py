"""Visualisasi hasil validasi independen model SST vs Argo float BMKG.

Menghasilkan 4 plot untuk BAB IV:
  1. Scatter plot: Model vs Argo SST (semua profil, warna per kelompok float)
  2. Time series: Domain average model vs observasi Argo harian
  3. Peta sebaran bias per lokasi float
  4. Bar chart: RMSE dan Pearson r per float

Output: reports/figures/argo_validation_*.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import yaml

# ── Konfigurasi ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def assign_group(bmkg_id: str) -> str:
    """Kelompokkan float berdasarkan tahun deployment."""
    if str(bmkg_id).startswith('FIBNDA2406') or str(bmkg_id) == '-':
        return 'Generasi 2024 (r≈0.97)'
    elif any(str(bmkg_id).startswith(p) for p in ['FIBNDA2211', 'FIFLRS', 'FIMKSR']):
        return 'Generasi Nov-2022/2023 (r≈0.77)'
    elif any(str(bmkg_id).startswith(p) for p in ['FIBNDA2203', 'FIWBND2203']):
        return 'Generasi Mar-2022 (r≈0.20)'
    return 'Lainnya'


GROUP_COLORS = {
    'Generasi 2024 (r≈0.97)'          : '#1D9E75',
    'Generasi Nov-2022/2023 (r≈0.77)' : '#378ADD',
    'Generasi Mar-2022 (r≈0.20)'      : '#E05C3A',
    'Lainnya'                          : '#888888',
}


def main():
    cfg      = load_config('configs/config.yaml')
    proc_dir = Path(cfg['data']['processed_dir'])
    fig_dir  = Path(cfg['evaluation']['figures_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Muat data ─────────────────────────────────────────────────────────────
    detail = pd.read_csv(Path(cfg['evaluation']['results_dir'])
                         / 'argo_validation_detail.csv')
    detail['date'] = pd.to_datetime(detail['date'])
    detail['group'] = detail['bmkg_id'].apply(assign_group)

    sst = pd.read_csv(proc_dir / 'sst_series.csv')
    sst['date'] = pd.to_datetime(sst['date'])

    print(f"Detail records : {len(detail):,}")
    print(f"SST series     : {len(sst):,} hari")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 1 — Scatter: Model vs Argo SST
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(7, 6))

    for group, grp in detail.groupby('group'):
        ax.scatter(grp['sst_argo'], grp['sst_model'],
                   c=GROUP_COLORS.get(group, '#888'),
                   alpha=0.55, s=18, label=group, edgecolors='none')

    # Garis 1:1
    smin = min(detail['sst_argo'].min(), detail['sst_model'].min()) - 0.5
    smax = max(detail['sst_argo'].max(), detail['sst_model'].max()) + 0.5
    ax.plot([smin, smax], [smin, smax], 'k--', lw=1.2, alpha=0.6, label='1:1 line')

    # Statistik keseluruhan
    n    = len(detail)
    bias = float(np.mean(detail['sst_model'] - detail['sst_argo']))
    rmse = float(np.sqrt(np.mean((detail['sst_model'] - detail['sst_argo'])**2)))
    r    = float(np.corrcoef(detail['sst_argo'], detail['sst_model'])[0, 1])

    stats_txt = f'n={n}\nBias={bias:+.3f}°C\nRMSE={rmse:.3f}°C\nr={r:.3f}'
    ax.text(0.04, 0.96, stats_txt, transform=ax.transAxes,
            va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8))

    ax.set_xlim(smin, smax)
    ax.set_ylim(smin, smax)
    ax.set_xlabel('SST Observasi Argo Float (°C)')
    ax.set_ylabel('SST Prediksi Model — Domain Average (°C)')
    ax.set_title('Validasi Independen: Model SST vs Observasi Argo Float BMKG\nLaut Banda, 2024–2025')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)

    out = fig_dir / 'argo_validation_scatter.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Scatter plot   : {out}")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 2 — Time Series: Model vs Argo (rata-rata harian)
    # ══════════════════════════════════════════════════════════════════════════
    daily_argo = (detail.groupby('date')['sst_argo']
                        .mean()
                        .reset_index()
                        .rename(columns={'sst_argo': 'sst_argo_mean'}))

    merged = pd.merge(daily_argo, sst, on='date', how='inner')
    merged = merged.sort_values('date')

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                              gridspec_kw={'height_ratios': [3, 1]})

    ax1 = axes[0]
    ax1.plot(merged['date'], merged['sst'],
             color='#378ADD', lw=1.5, label='Model (domain avg)', zorder=2)
    ax1.scatter(merged['date'], merged['sst_argo_mean'],
                color='#E05C3A', s=20, zorder=3,
                label='Argo float (rata-rata harian)', alpha=0.8)
    ax1.set_ylabel('SST (°C)')
    ax1.set_title('Time Series SST: Model Domain Average vs Observasi Argo Float\nLaut Banda, 2024–2025')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.set_xlim(merged['date'].min(), merged['date'].max())

    # Panel residual
    ax2 = axes[1]
    resid = merged['sst'] - merged['sst_argo_mean']
    ax2.bar(merged['date'], resid,
            color=np.where(resid >= 0, '#378ADD', '#E05C3A'),
            alpha=0.7, width=2)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axhline(resid.mean(), color='purple', lw=1, ls='--',
                label=f'Rata-rata bias={resid.mean():+.3f}°C')
    ax2.set_ylabel('Residual (°C)')
    ax2.set_xlabel('Tanggal')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    ax2.set_xlim(merged['date'].min(), merged['date'].max())

    fig.tight_layout()
    out = fig_dir / 'argo_validation_timeseries.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Time series    : {out}")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 3 — Bar chart: RMSE dan r per float
    # ══════════════════════════════════════════════════════════════════════════
    per_float = []
    for (imei, bmkg), grp in detail.groupby(['imei', 'bmkg_id']):
        obs  = grp['sst_argo'].values
        pred = grp['sst_model'].values
        rmse_f = float(np.sqrt(np.mean((pred - obs)**2)))
        r_f    = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 2 else 0.0
        bias_f = float(np.mean(pred - obs))
        group  = grp['group'].iloc[0]
        label  = bmkg if str(bmkg) != '-' else f'Unknown-{str(imei)[-4:]}'
        per_float.append(dict(label=label, rmse=rmse_f, r=r_f,
                               bias=bias_f, n=len(grp), group=group))

    pf = pd.DataFrame(per_float).sort_values('r', ascending=True)

    fig, (ax_rmse, ax_r) = plt.subplots(1, 2, figsize=(13, 6))
    colors = [GROUP_COLORS.get(g, '#888') for g in pf['group']]

    # RMSE bars
    bars = ax_rmse.barh(pf['label'], pf['rmse'], color=colors, alpha=0.85)
    ax_rmse.axvline(pf['rmse'].mean(), color='black', lw=1.2, ls='--',
                    label=f'Rata-rata={pf["rmse"].mean():.3f}°C')
    ax_rmse.set_xlabel('RMSE (°C)')
    ax_rmse.set_title('RMSE per Float')
    ax_rmse.legend(fontsize=8)
    ax_rmse.grid(alpha=0.25, axis='x')
    for bar, n in zip(bars, pf['n']):
        ax_rmse.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f'n={n}', va='center', fontsize=7.5)

    # Pearson r bars
    bars2 = ax_r.barh(pf['label'], pf['r'], color=colors, alpha=0.85)
    ax_r.axvline(pf['r'].mean(), color='black', lw=1.2, ls='--',
                 label=f'Rata-rata r={pf["r"].mean():.3f}')
    ax_r.axvline(0.7, color='green', lw=1, ls=':', alpha=0.7,
                 label='Threshold r=0.7')
    ax_r.set_xlabel('Pearson r')
    ax_r.set_title('Korelasi Pearson per Float')
    ax_r.set_xlim(-0.1, 1.05)
    ax_r.legend(fontsize=8)
    ax_r.grid(alpha=0.25, axis='x')

    # Legend kelompok
    legend_elements = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=c, markersize=10, label=g)
        for g, c in GROUP_COLORS.items() if g != 'Lainnya'
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Performa Validasi per Float Argo BMKG — Laut Banda 2024–2025',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = fig_dir / 'argo_validation_per_float.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Per-float chart: {out}")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 4 — Peta sebaran bias per lokasi rata-rata float
    # ══════════════════════════════════════════════════════════════════════════
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        float_locs = (detail.groupby(['imei', 'bmkg_id'])
                            .agg(lon=('lon', 'mean'),
                                 lat=('lat', 'mean'),
                                 bias=('sst_model', lambda x:
                                       (x - detail.loc[x.index, 'sst_argo']).mean()),
                                 r=('sst_argo', lambda x:
                                    np.corrcoef(x,
                                    detail.loc[x.index, 'sst_model'])[0,1]
                                    if len(x) > 2 else 0),
                                 n=('sst_argo', 'count'))
                            .reset_index())

        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(1, 1, 1,
                               projection=ccrs.PlateCarree())
        ax.set_extent([122.0, 134.0, -10.0, -2.0], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#E8E8E8', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Plot titik bias
        sc = ax.scatter(float_locs['lon'], float_locs['lat'],
                        c=float_locs['bias'],
                        cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                        s=float_locs['n'] * 1.5,
                        transform=ccrs.PlateCarree(),
                        edgecolors='black', linewidths=0.5, zorder=3,
                        alpha=0.9)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label('Bias Model–Argo (°C)', fontsize=9)

        # Annotasi nama float
        for _, row in float_locs.iterrows():
            label = row['bmkg_id'] if str(row['bmkg_id']) != '-' \
                    else f"Unk-{str(row['imei'])[-4:]}"
            ax.text(row['lon'] + 0.15, row['lat'] + 0.1, label,
                    fontsize=6, transform=ccrs.PlateCarree(),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              alpha=0.6, linewidth=0))

        # Legend ukuran simbol
        for ns, ls in [(20, 'n=20'), (60, 'n=60'), (100, 'n=100')]:
            ax.scatter([], [], s=ns*1.5, c='gray', alpha=0.5,
                       label=ls, transform=ccrs.PlateCarree())
        ax.legend(title='Jumlah profil', fontsize=7,
                  loc='lower right', framealpha=0.8)

        ax.set_title('Sebaran Spasial Bias Validasi Model vs Argo Float BMKG\n'
                     'Laut Banda, 2024–2025 (ukuran simbol ∝ jumlah profil)',
                     fontsize=10)

        out = fig_dir / 'argo_validation_bias_map.png'
        fig.savefig(out)
        plt.close(fig)
        print(f"Bias map       : {out}")

    except Exception as e:
        print(f"Bias map skip  : {e}")

    print(f"\n✓ Semua visualisasi tersimpan di {fig_dir}/")


if __name__ == '__main__':
    main()
