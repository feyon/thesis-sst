from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

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

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def assign_group(bmkg_id):
    s = str(bmkg_id)
    if s.startswith('FIBNDA2406') or s.startswith('FIBNDA2409'):
        return 'Generasi 2024 (r≈0.97)'
    elif any(s.startswith(p) for p in ['FIBNDA2211','FIFLRS','FIMKSR']):
        return 'Generasi Nov-2022/2023 (r≈0.80)'
    elif any(s.startswith(p) for p in ['FIBNDA2203','FIWBND2203']):
        return 'Generasi Mar-2022 (r≈0.20)'
    return 'Lainnya'

GROUP_COLORS = {
    'Generasi 2024 (r≈0.97)'         : '#1D9E75',
    'Generasi Nov-2022/2023 (r≈0.80)': '#378ADD',
    'Generasi Mar-2022 (r≈0.20)'     : '#E05C3A',
    'Lainnya'                         : '#888888',
}

def metrics(obs, pred):
    bias = float(np.mean(pred - obs))
    rmse = float(np.sqrt(np.mean((pred - obs)**2)))
    mae  = float(np.mean(np.abs(pred - obs)))
    r    = float(np.corrcoef(obs, pred)[0,1]) if len(obs) > 2 else 0.0
    return bias, rmse, mae, r

def main():
    cfg     = load_config('configs/config.yaml')
    fig_dir = Path(cfg['evaluation']['figures_dir'])
    res_dir = Path(cfg['evaluation']['results_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(res_dir / 'argo_collocation_detail.csv')
    detail['date']  = pd.to_datetime(detail['date'])
    detail['group'] = detail['bmkg_id'].apply(assign_group)

    print(f"Records: {len(detail):,}")

    # ── PLOT 1: Scatter Model vs Argo ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for group, grp in detail.groupby('group'):
        ax.scatter(grp['sst_argo'], grp['sst_cmems'],
                   c=GROUP_COLORS.get(group,'#888'),
                   alpha=0.55, s=18, label=group, edgecolors='none')

    smin = min(detail['sst_argo'].min(), detail['sst_cmems'].min()) - 0.5
    smax = max(detail['sst_argo'].max(), detail['sst_cmems'].max()) + 0.5
    ax.plot([smin,smax],[smin,smax],'k--',lw=1.2,alpha=0.6,label='1:1 line')

    bias,rmse,mae,r = metrics(detail['sst_argo'].values,
                               detail['sst_cmems'].values)
    ax.text(0.04, 0.96,
            f'n={len(detail)}\nBias={bias:+.3f}°C\nRMSE={rmse:.3f}°C\nr={r:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4',fc='white',alpha=0.8))
    ax.set_xlim(smin,smax); ax.set_ylim(smin,smax)
    ax.set_xlabel('SST Observasi Argo Float (°C)')
    ax.set_ylabel('SST CMEMS di Titik Terdekat (°C)')
    ax.set_title('Validasi Spatial Collocation KDTree\nModel SST vs Argo Float BMKG — Laut Banda 2024–2025')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)
    fig.savefig(fig_dir / 'argo_collocation_scatter.png')
    plt.close(fig)
    print(f"Scatter    : {fig_dir}/argo_collocation_scatter.png")

    # ── PLOT 2: Time Series ───────────────────────────────────────────────────
    daily = (detail.groupby('date')
                   .agg(sst_argo_mean=('sst_argo','mean'),
                        sst_cmems_mean=('sst_cmems','mean'))
                   .reset_index().sort_values('date'))

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,7),
                                   gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(daily['date'], daily['sst_cmems_mean'],
             color='#378ADD', lw=1.5, label='CMEMS di titik Argo', zorder=2)
    ax1.scatter(daily['date'], daily['sst_argo_mean'],
                color='#E05C3A', s=20, zorder=3,
                label='Argo float (rata-rata harian)', alpha=0.8)
    ax1.set_ylabel('SST (°C)')
    ax1.set_title('Time Series SST: CMEMS vs Observasi Argo Float\nLaut Banda, 2024–2025 (Spatial Collocation KDTree)')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)
    ax1.set_xlim(daily['date'].min(), daily['date'].max())

    resid = daily['sst_cmems_mean'] - daily['sst_argo_mean']
    ax2.bar(daily['date'], resid,
            color=np.where(resid>=0,'#378ADD','#E05C3A'),
            alpha=0.7, width=2)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axhline(resid.mean(), color='purple', lw=1, ls='--',
                label=f'Rata-rata bias={resid.mean():+.3f}°C')
    ax2.set_ylabel('Residual (°C)')
    ax2.set_xlabel('Tanggal')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    ax2.set_xlim(daily['date'].min(), daily['date'].max())
    fig.tight_layout()
    fig.savefig(fig_dir / 'argo_collocation_timeseries.png')
    plt.close(fig)
    print(f"Time series: {fig_dir}/argo_collocation_timeseries.png")

    # ── PLOT 3: Bar chart RMSE & r per float ─────────────────────────────────
    per_float = []
    for bmkg, grp in detail.groupby('bmkg_id'):
        b,rm,ma,r = metrics(grp['sst_argo'].values, grp['sst_cmems'].values)
        per_float.append(dict(label=str(bmkg), rmse=rm, r=r,
                               bias=b, n=len(grp),
                               group=grp['group'].iloc[0]))
    pf = pd.DataFrame(per_float).sort_values('r', ascending=True)
    colors = [GROUP_COLORS.get(g,'#888') for g in pf['group']]

    fig, (ax_r, ax_rmse) = plt.subplots(1,2,figsize=(13,6))

    bars = ax_rmse.barh(pf['label'], pf['rmse'], color=colors, alpha=0.85)
    ax_rmse.axvline(pf['rmse'].mean(), color='black', lw=1.2, ls='--',
                    label=f'Rata-rata={pf["rmse"].mean():.3f}°C')
    ax_rmse.set_xlabel('RMSE (°C)')
    ax_rmse.set_title('RMSE per Float')
    ax_rmse.legend(fontsize=8); ax_rmse.grid(alpha=0.25, axis='x')
    for bar, n in zip(bars, pf['n']):
        ax_rmse.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                     f'n={n}', va='center', fontsize=7.5)

    bars2 = ax_r.barh(pf['label'], pf['r'], color=colors, alpha=0.85)
    ax_r.axvline(pf['r'].mean(), color='black', lw=1.2, ls='--',
                 label=f'Rata-rata r={pf["r"].mean():.3f}')
    ax_r.axvline(0.7, color='green', lw=1, ls=':', alpha=0.7,
                 label='Threshold r=0.7')
    ax_r.set_xlabel('Pearson r')
    ax_r.set_title('Korelasi Pearson per Float')
    ax_r.set_xlim(-0.1,1.05)
    ax_r.legend(fontsize=8); ax_r.grid(alpha=0.25, axis='x')

    legend_elements = [
        Line2D([0],[0],marker='s',color='w',
               markerfacecolor=c,markersize=10,label=g)
        for g,c in GROUP_COLORS.items() if g != 'Lainnya'
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5,-0.02))
    fig.suptitle('Performa Validasi Spatial Collocation per Float Argo BMKG\nLaut Banda 2024–2025',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0.06,1,1])
    fig.savefig(fig_dir / 'argo_collocation_per_float.png')
    plt.close(fig)
    print(f"Per-float  : {fig_dir}/argo_collocation_per_float.png")

    # ── PLOT 4: Bias Map ──────────────────────────────────────────────────────
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        float_locs = (detail.groupby('bmkg_id')
                            .agg(lon=('lon','mean'),
                                 lat=('lat','mean'),
                                 bias=('bias','mean'),
                                 n=('sst_argo','count'))
                            .reset_index())
        float_locs['r'] = float_locs['bmkg_id'].map(
            detail.groupby('bmkg_id').apply(
                lambda g: float(np.corrcoef(g['sst_argo'],g['sst_cmems'])[0,1])
                if len(g)>2 else 0.0
            )
        )

        fig = plt.figure(figsize=(10,7))
        ax  = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        ax.set_extent([122.0,134.0,-10.0,-2.0],crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND,facecolor='#E8E8E8',zorder=1)
        ax.add_feature(cfeature.COASTLINE,linewidth=0.5,zorder=2)
        ax.gridlines(draw_labels=True,alpha=0.3,linestyle='--',linewidth=0.5)

        sc = ax.scatter(float_locs['lon'], float_locs['lat'],
                        c=float_locs['bias'],
                        cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                        s=float_locs['n']*1.5,
                        transform=ccrs.PlateCarree(),
                        edgecolors='black',linewidths=0.5,zorder=3,alpha=0.9)
        cbar = plt.colorbar(sc,ax=ax,shrink=0.6,pad=0.08)
        cbar.set_label('Bias CMEMS–Argo (°C)',fontsize=9)

        for _,row in float_locs.iterrows():
            ax.text(row['lon']+0.15, row['lat']+0.1, str(row['bmkg_id']),
                    fontsize=6, transform=ccrs.PlateCarree(),
                    bbox=dict(boxstyle='round,pad=0.2',fc='white',
                              alpha=0.6,linewidth=0))

        for ns,ls in [(20,'n=20'),(60,'n=60'),(100,'n=100')]:
            ax.scatter([],[],s=ns*1.5,c='gray',alpha=0.5,label=ls,
                       transform=ccrs.PlateCarree())
        ax.legend(title='Jumlah profil',fontsize=7,
                  loc='lower right',framealpha=0.8)
        ax.set_title('Sebaran Spasial Bias Validasi CMEMS vs Argo Float BMKG\n'
                     'Laut Banda 2024–2025 (Spatial Collocation KDTree)',fontsize=10)
        fig.savefig(fig_dir / 'argo_collocation_bias_map.png')
        plt.close(fig)
        print(f"Bias map   : {fig_dir}/argo_collocation_bias_map.png")
    except Exception as e:
        print(f"Bias map skip: {e}")

    print(f"\n✓ Semua visualisasi tersimpan di {fig_dir}/")

if __name__ == '__main__':
    main()
