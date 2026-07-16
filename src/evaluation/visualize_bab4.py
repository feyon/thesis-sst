"""Visualisasi tambahan untuk BAB IV.

Menghasilkan:
  1. Heatmap RMSE dan Skill Score (model x horizon, per lookback)
  2. Plot degradasi akurasi — RMSE vs horizon per model
  3. Time series prediksi model terbaik vs SST aktual (test set 2024-2025)
  4. Analisis error musiman SST Laut Banda

Output: results/figures/bab4_*.png
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size'  : 10,
    'axes.titlesize': 11,
    'figure.dpi' : 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

MODEL_LABELS  = {'lstm': 'LSTM', 'transformer': 'Transformer', 'hybrid': 'Hybrid'}
MODEL_COLORS  = {'lstm': '#378ADD', 'transformer': '#EF9F27', 'hybrid': '#1D9E75'}
HORIZONS      = [1, 3, 7, 14]
LOOKBACKS     = [7, 14, 21, 30]
MODELS        = ['lstm', 'transformer', 'hybrid']


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_results(res_dir: Path) -> pd.DataFrame:
    """Muat semua 48 JSON hasil training ke DataFrame."""
    records = []
    for f in res_dir.glob('*.json'):
        if 'argo' in f.name or 'summary' in f.name:
            continue
        try:
            d = json.loads(f.read_text())
            records.append(d)
        except Exception:
            pass
    df = pd.DataFrame(records)
    df['model']    = df['model'].str.lower()
    df['lookback'] = df['lookback'].astype(int)
    df['horizon']  = df['horizon'].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Heatmap RMSE dan Skill Score
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmaps(df: pd.DataFrame, fig_dir: Path) -> None:
    for metric, cmap, label, fmt in [
        ('rmse',        'YlOrRd',  'RMSE (°C)',    '.3f'),
        ('skill_score', 'RdYlGn',  'Skill Score',  '.3f'),
        ('r2',          'YlGn',    'R²',           '.3f'),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Heatmap {label} — Model × Horizon × Lookback',
                     fontsize=12, fontweight='bold', y=1.02)

        for ax, model in zip(axes, MODELS):
            sub = df[df['model'] == model].copy()
            # Pivot: baris=lookback, kolom=horizon
            pivot = sub.pivot(index='lookback', columns='horizon',
                              values=metric).reindex(
                index=LOOKBACKS, columns=HORIZONS)

            if metric == 'skill_score':
                vmin, vcenter, vmax = -0.1, 0.3, 0.5
                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                im = ax.imshow(pivot.values, cmap=cmap, norm=norm,
                               aspect='auto')
            else:
                im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')

            plt.colorbar(im, ax=ax, shrink=0.8, label=label)

            ax.set_xticks(range(len(HORIZONS)))
            ax.set_yticks(range(len(LOOKBACKS)))
            ax.set_xticklabels([f'h={h}' for h in HORIZONS])
            ax.set_yticklabels([f'lb={lb}' for lb in LOOKBACKS])
            ax.set_xlabel('Horizon Prediksi (hari)')
            ax.set_ylabel('Lookback Window (hari)')
            ax.set_title(MODEL_LABELS[model])

            # Anotasi nilai di setiap sel
            for i in range(len(LOOKBACKS)):
                for j in range(len(HORIZONS)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        color = 'white' if metric == 'rmse' and val > 0.35 \
                                else 'black'
                        ax.text(j, i, f'{val:{fmt}}',
                                ha='center', va='center',
                                fontsize=8.5, color=color, fontweight='bold')

            # Tandai sel terbaik
            if metric == 'skill_score':
                best_idx = np.unravel_index(
                    np.nanargmax(pivot.values), pivot.values.shape)
            else:
                best_idx = np.unravel_index(
                    np.nanargmin(pivot.values), pivot.values.shape)
            ax.add_patch(plt.Rectangle(
                (best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                fill=False, edgecolor='blue', linewidth=2.5))

        fig.tight_layout()
        out = fig_dir / f'bab4_heatmap_{metric}.png'
        fig.savefig(out)
        plt.close(fig)
        print(f"Heatmap {metric:<12}: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Degradasi Akurasi (RMSE vs Horizon)
# ─────────────────────────────────────────────────────────────────────────────
def plot_degradasi(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, ylabel, title in [
        (axes[0], 'rmse',        'RMSE (°C)',   'Degradasi RMSE vs Horizon Prediksi'),
        (axes[1], 'skill_score', 'Skill Score', 'Skill Score vs Horizon Prediksi'),
    ]:
        for model in MODELS:
            sub = df[df['model'] == model]
            # Rata-rata semua lookback per horizon
            mean = sub.groupby('horizon')[metric].mean()
            std  = sub.groupby('horizon')[metric].std()
            ax.plot(mean.index, mean.values,
                    color=MODEL_COLORS[model], lw=2,
                    marker='o', ms=7, label=MODEL_LABELS[model])
            ax.fill_between(mean.index,
                            mean.values - std.values,
                            mean.values + std.values,
                            alpha=0.12, color=MODEL_COLORS[model])

        if metric == 'skill_score':
            ax.axhline(0.3, color='red', lw=1.2, ls='--', alpha=0.7,
                       label='Threshold SS=0.3')
            ax.axhline(0.0, color='gray', lw=0.8, ls=':', alpha=0.5,
                       label='Batas minimal (SS=0)')

        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([f'h={h}d' for h in HORIZONS])
        ax.set_xlabel('Horizon Prediksi (hari)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle('Analisis Degradasi Akurasi Model SST — Laut Banda 2024–2025',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = fig_dir / 'bab4_degradasi_akurasi.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Degradasi akurasi  : {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Time Series Prediksi Terbaik vs Aktual
# ─────────────────────────────────────────────────────────────────────────────
def plot_timeseries_prediksi(cfg: dict, fig_dir: Path,
                              ckpt_dir: Path, proc_dir: Path) -> None:
    """Muat checkpoint hybrid_lb30_h14, prediksi test set, plot vs aktual."""
    from src.data.dataset import build_loaders
    from src.models.factory import build_model
    from src.training.trainer import predict

    ckpt_path = ckpt_dir / 'hybrid_lb30_h14.pt'
    if not ckpt_path.exists():
        print(f"Checkpoint tidak ditemukan: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = build_model('hybrid', ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    df = pd.read_csv(proc_dir / 'sst_series.csv', parse_dates=['date'])
    data = build_loaders(df, cfg, lookback=30, horizon=14)

    scaler = data['scaler']
    Xte, yte = data['arrays']['test']

    y_pred_norm = predict(model, Xte, ckpt['config'])
    y_true  = scaler.inverse(yte)
    y_pred  = scaler.inverse(y_pred_norm)

    # Rekonstruksi tanggal test set
    sp = cfg['split']
    test_start = pd.Timestamp(sp['test'][0]) + pd.Timedelta(days=30+14-1)
    dates = pd.date_range(test_start, periods=len(y_true), freq='D')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={'height_ratios': [3, 1, 1]})

    # Panel atas: SST aktual vs prediksi
    ax1 = axes[0]
    ax1.plot(dates, y_true,  color='#333333', lw=1.2,
             label='SST Aktual (CMEMS)', alpha=0.9)
    ax1.plot(dates, y_pred,  color='#1D9E75', lw=1.5, ls='--',
             label='Prediksi Hybrid lb30 h14', alpha=0.9)
    ax1.set_ylabel('SST (°C)')
    ax1.set_title('Prediksi SST Model Hybrid LSTM-Transformer Terbaik\n'
                  'hybrid_lb30_h14 vs SST Aktual — Test Set 2024–2025')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)
    ax1.set_xlim(dates[0], dates[-1])

    # Panel tengah: residual
    ax2 = axes[1]
    resid = y_pred - y_true
    ax2.bar(dates, resid,
            color=np.where(resid >= 0, '#1D9E75', '#E05C3A'),
            alpha=0.7, width=1)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axhline(resid.mean(), color='purple', lw=1, ls='--',
                label=f'Rata-rata bias={resid.mean():+.4f}°C')
    ax2.set_ylabel('Residual (°C)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)
    ax2.set_xlim(dates[0], dates[-1])

    # Panel bawah: rolling RMSE 30 hari
    ax3 = axes[2]
    roll_rmse = pd.Series(resid**2).rolling(30).mean().apply(np.sqrt)
    ax3.plot(dates, roll_rmse.values, color='#E05C3A', lw=1.5)
    ax3.axhline(np.sqrt(np.mean(resid**2)), color='black', lw=1, ls='--',
                label=f'RMSE keseluruhan={np.sqrt(np.mean(resid**2)):.4f}°C')
    ax3.set_ylabel('Rolling RMSE 30d (°C)')
    ax3.set_xlabel('Tanggal')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)
    ax3.set_xlim(dates[0], dates[-1])

    # Statistik di panel atas
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    r    = float(np.corrcoef(y_true, y_pred)[0, 1])
    ax1.text(0.01, 0.97,
             f'RMSE={rmse:.4f}°C  MAE={mae:.4f}°C  r={r:.4f}',
             transform=ax1.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

    fig.tight_layout()
    out = fig_dir / 'bab4_timeseries_prediksi_terbaik.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Time series prediksi: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Analisis Error Musiman
# ─────────────────────────────────────────────────────────────────────────────
def plot_musiman(df_results: pd.DataFrame, cfg: dict,
                 fig_dir: Path, ckpt_dir: Path, proc_dir: Path) -> None:
    """Analisis error per bulan untuk model terbaik (hybrid_lb30_h14)."""
    from src.data.dataset import build_loaders
    from src.models.factory import build_model
    from src.training.trainer import predict

    ckpt_path = ckpt_dir / 'hybrid_lb30_h14.pt'
    if not ckpt_path.exists():
        print(f"Checkpoint tidak ditemukan: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = build_model('hybrid', ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    df = pd.read_csv(proc_dir / 'sst_series.csv', parse_dates=['date'])
    data = build_loaders(df, cfg, lookback=30, horizon=14)
    scaler = data['scaler']
    Xte, yte = data['arrays']['test']

    y_pred_norm = predict(model, Xte, ckpt['config'])
    y_true = scaler.inverse(yte)
    y_pred = scaler.inverse(y_pred_norm)

    sp = cfg['split']
    test_start = pd.Timestamp(sp['test'][0]) + pd.Timedelta(days=30+14-1)
    dates = pd.date_range(test_start, periods=len(y_true), freq='D')

    ts = pd.DataFrame({
        'date'  : dates,
        'y_true': y_true,
        'y_pred': y_pred,
        'resid' : y_pred - y_true,
        'month' : dates.month,
        'season': dates.month.map({
            12: 'Des-Feb\n(Monsun Barat)', 1: 'Des-Feb\n(Monsun Barat)',
            2:  'Des-Feb\n(Monsun Barat)',
            3:  'Mar-Mei\n(Peralihan I)',  4: 'Mar-Mei\n(Peralihan I)',
            5:  'Mar-Mei\n(Peralihan I)',
            6:  'Jun-Agt\n(Monsun Timur)', 7: 'Jun-Agt\n(Monsun Timur)',
            8:  'Jun-Agt\n(Monsun Timur)',
            9:  'Sep-Nov\n(Peralihan II)', 10:'Sep-Nov\n(Peralihan II)',
            11: 'Sep-Nov\n(Peralihan II)',
        })
    })

    SEASON_ORDER = ['Des-Feb\n(Monsun Barat)', 'Mar-Mei\n(Peralihan I)',
                    'Jun-Agt\n(Monsun Timur)', 'Sep-Nov\n(Peralihan II)']
    SEASON_COLORS = ['#378ADD', '#1D9E75', '#EF9F27', '#E05C3A']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: SST rata-rata per bulan
    monthly_sst = ts.groupby('month')['y_true'].mean()
    axes[0].bar(monthly_sst.index, monthly_sst.values,
                color='#378ADD', alpha=0.8, edgecolor='white')
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(['Jan','Feb','Mar','Apr','Mei','Jun',
                              'Jul','Agt','Sep','Okt','Nov','Des'],
                             fontsize=8)
    axes[0].set_xlabel('Bulan')
    axes[0].set_ylabel('SST Rata-rata (°C)')
    axes[0].set_title('Variabilitas SST Bulanan\nLaut Banda 2024–2025')
    axes[0].grid(alpha=0.25, axis='y')

    # Panel 2: RMSE per bulan
    monthly_rmse = ts.groupby('month')['resid'].apply(
        lambda x: np.sqrt(np.mean(x**2)))
    axes[1].bar(monthly_rmse.index, monthly_rmse.values,
                color='#E05C3A', alpha=0.8, edgecolor='white')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','Mei','Jun',
                              'Jul','Agt','Sep','Okt','Nov','Des'],
                             fontsize=8)
    axes[1].set_xlabel('Bulan')
    axes[1].set_ylabel('RMSE (°C)')
    axes[1].set_title('RMSE Bulanan Model Hybrid lb30 h14\nLaut Banda 2024–2025')
    axes[1].grid(alpha=0.25, axis='y')

    # Panel 3: Boxplot bias per musim
    season_data = [ts[ts['season'] == s]['resid'].values
                   for s in SEASON_ORDER]
    bp = axes[2].boxplot(season_data, patch_artist=True,
                          medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], SEASON_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[2].axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    axes[2].set_xticklabels(SEASON_ORDER, fontsize=8)
    axes[2].set_xlabel('Musim')
    axes[2].set_ylabel('Residual / Bias (°C)')
    axes[2].set_title('Distribusi Bias per Musim\nMonsun Laut Banda')
    axes[2].grid(alpha=0.25, axis='y')

    # Anotasi RMSE per musim
    for i, (s, color) in enumerate(zip(SEASON_ORDER, SEASON_COLORS)):
        vals = ts[ts['season'] == s]['resid'].values
        rmse_s = np.sqrt(np.mean(vals**2))
        axes[2].text(i+1, axes[2].get_ylim()[1]*0.92,
                     f'RMSE\n{rmse_s:.3f}°C',
                     ha='center', fontsize=7.5, color=color, fontweight='bold')

    fig.suptitle('Analisis Error Musiman Model Hybrid LSTM-Transformer\nLaut Banda 2024–2025',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = fig_dir / 'bab4_analisis_musiman.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Analisis musiman    : {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import sys
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    args = p.parse_args()

    cfg      = load_config(args.config)
    fig_dir  = Path(cfg['evaluation']['figures_dir'])
    res_dir  = Path(cfg['evaluation']['results_dir'])
    ckpt_dir = Path('models/checkpoints')
    proc_dir = Path(cfg['data']['processed_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Muat semua hasil
    df = load_results(res_dir)
    print(f"Hasil dimuat: {len(df)} run")
    print(f"Model: {sorted(df['model'].unique())}")
    print(f"Horizons: {sorted(df['horizon'].unique())}")
    print()

    # Plot 1 — Heatmap
    plot_heatmaps(df, fig_dir)

    # Plot 2 — Degradasi akurasi
    plot_degradasi(df, fig_dir)

    # Plot 3 — Time series prediksi terbaik
    plot_timeseries_prediksi(cfg, fig_dir, ckpt_dir, proc_dir)

    # Plot 4 — Analisis musiman
    plot_musiman(df, cfg, fig_dir, ckpt_dir, proc_dir)

    print(f"\n✓ Semua visualisasi BAB IV tersimpan di {fig_dir}/")


if __name__ == '__main__':
    main()
