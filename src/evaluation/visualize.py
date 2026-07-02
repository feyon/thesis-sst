"""Visualisasi analisis hasil dari results/results_all.csv."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -", path)


def plot_rmse_vs_horizon(res, figdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    best_lb = res.groupby("model")["rmse"].idxmin()
    for model in res["model"].unique():
        sub = (res[res.model == model]
               .groupby("horizon")["rmse"].min().reset_index())
        ax.plot(sub["horizon"], sub["rmse"], marker="o", label=model)
    ax.set_xlabel("Horizon prediksi (hari)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("RMSE vs Horizon (lookback terbaik per model)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, figdir / "rmse_vs_horizon.png")


def plot_skillscore_bar(res, figdir, threshold=0.3):
    dl = res[res.model != "persistence"]
    pivot = dl.pivot_table(index="model", values="skill_score", aggfunc="max")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(pivot.index, pivot["skill_score"])
    ax.axhline(threshold, color="red", ls="--", label=f"gate {threshold}")
    ax.set_ylabel("Skill Score (maks) vs Persistence")
    ax.set_title("Skill Score terbaik per model")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    _save(fig, figdir / "skillscore_bar.png")


def plot_heatmap(res, figdir, model="hybrid", metric="rmse"):
    sub = res[res.model == model]
    if sub.empty:
        return
    grid = sub.pivot(index="lookback", columns="horizon", values=metric)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(grid.columns))); ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index))); ax.set_yticklabels(grid.index)
    ax.set_xlabel("Horizon (hari)"); ax.set_ylabel("Lookback (hari)")
    ax.set_title(f"{metric.upper()} — model {model}")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label=metric.upper())
    _save(fig, figdir / f"heatmap_{model}_{metric}.png")


def main(cfg, results_csv):
    res = pd.read_csv(results_csv)
    figdir = Path(cfg["evaluation"]["figures_dir"])
    thr = cfg["evaluation"]["skill_score_threshold"]
    print("Membuat figur analisis:")
    plot_rmse_vs_horizon(res, figdir)
    plot_skillscore_bar(res, figdir, thr)
    plot_heatmap(res, figdir, model="hybrid", metric="rmse")

    # ringkasan tabel: model terbaik per horizon (berdasar RMSE)
    summary = (res.loc[res.groupby("horizon")["rmse"].idxmin()]
               [["horizon", "model", "lookback", "rmse", "skill_score"]])
    summary.to_csv(Path(cfg["evaluation"]["results_dir"]) / "summary_best.csv",
                   index=False)
    print("\nRingkasan model terbaik per horizon:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--results", default="results/results_all.csv")
    a = p.parse_args()
    main(load_config(a.config), a.results)
