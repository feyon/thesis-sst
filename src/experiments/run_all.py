"""
Jalankan SELURUH grid eksperimen (experiment.lookback_windows x
experiment.horizons x experiment.models) secara berurutan: train lalu
evaluate tiap kombinasi, kumpulkan semua metrics_overall.csv jadi satu
tabel perbandingan untuk memilih kombinasi terbaik.

Setiap kombinasi dijalankan sbg SUBPROCESS terpisah (memanggil
`python -m src.training.train` lalu `python -m src.evaluation.evaluate`)
supaya memori GPU dibebaskan bersih antar run, bukan menumpuk dalam satu
proses Python yang sama.

Cara jalan (SLURM + apptainer, GPU):
    cd ~/thesis-sst
    srun --partition=gpu_riset --gres=gpu:1 --cpus-per-task=4 --mem=16G \\
      --time=03:00:00 \\
      apptainer exec --nv --pwd /workspace \\
      --bind ~/thesis-sst/data:/workspace/data \\
      --bind ~/thesis-sst/configs:/workspace/configs \\
      --bind ~/thesis-sst/src:/workspace/src \\
      --bind ~/thesis-sst/results:/workspace/results \\
      ~/thesis-sst/thesis-sst.sif \\
      python -m src.experiments.run_all --config configs/config.yaml

Opsi berguna:
    --skip_existing   lewati kombinasi yang sudah py punya summary.json
                       (aman utk melanjutkan run yang terputus)
    --models, --lookbacks, --horizons
                       override subset grid, mis. utk tes cepat:
                       --models lstm --lookbacks 7 --horizons 1

Output: <results_dir>/evaluation/all_experiments_summary.csv
    Satu baris per kombinasi (model, lookback, horizon), berisi semua
    metrik overall + verdict lolos/belum lolos skill_score_threshold,
    diurutkan dari Skill Score tertinggi.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_subprocess(cmd, label):
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    GAGAL ({label}). stderr (500 char terakhir):")
        print("    " + result.stderr[-500:].replace("\n", "\n    "))
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--lookbacks", nargs="+", type=int, default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override training.max_epochs (mis. utk tes cepat)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg["experiment"]
    results_dir = REPO_ROOT / cfg["evaluation"]["results_dir"]

    models = args.models or exp_cfg["models"]
    lookbacks = args.lookbacks or exp_cfg["lookback_windows"]
    horizons = args.horizons or exp_cfg["horizons"]

    combos = [(m, lb, h) for m in models for lb in lookbacks for h in horizons]
    print(f"Total kombinasi: {len(combos)} "
          f"({len(models)} model x {len(lookbacks)} lookback x "
          f"{len(horizons)} horizon)\n")

    all_rows = []
    n_ok, n_skip, n_fail = 0, 0, 0

    for i, (model, lb, h) in enumerate(combos, 1):
        run_name = f"{model}_lb{lb:02d}_h{h:02d}"
        print(f"[{i}/{len(combos)}] {run_name}")

        ckpt_summary = results_dir / "checkpoints" / run_name / "summary.json"
        eval_csv = (results_dir / "evaluation" / f"{run_name}_{args.split}" /
                   "metrics_overall.csv")

        if args.skip_existing and ckpt_summary.exists() and eval_csv.exists():
            print("    sudah ada (skip_existing), pakai hasil lama.")
            all_rows.append(pd.read_csv(eval_csv).iloc[0].to_dict())
            n_skip += 1
            continue

        # ---- training ----
        if not (args.skip_existing and ckpt_summary.exists()):
            train_cmd = [
                sys.executable, "-m", "src.training.train",
                "--config", args.config,
                "--lookback", str(lb), "--horizon", str(h), "--model", model,
            ]
            if args.max_epochs:
                train_cmd += ["--max_epochs", str(args.max_epochs)]
            ok = run_subprocess(train_cmd, f"{run_name} [train]")
            if not ok:
                n_fail += 1
                continue
        else:
            print("    checkpoint sudah ada, skip training.")

        # ---- evaluasi ----
        eval_cmd = [
            sys.executable, "-m", "src.evaluation.evaluate",
            "--config", args.config,
            "--lookback", str(lb), "--horizon", str(h), "--model", model,
            "--split", args.split,
        ]
        ok = run_subprocess(eval_cmd, f"{run_name} [evaluate]")
        if not ok:
            n_fail += 1
            continue

        if eval_csv.exists():
            all_rows.append(pd.read_csv(eval_csv).iloc[0].to_dict())
            n_ok += 1
        else:
            print(f"    PERINGATAN: {eval_csv} tidak ditemukan setelah evaluate.")
            n_fail += 1

    # ------------------------------------------------------------------
    if not all_rows:
        print("\nTidak ada kombinasi yang berhasil sama sekali.")
        return

    summary_df = pd.DataFrame(all_rows)
    threshold = cfg["evaluation"]["skill_score_threshold"]
    summary_df["lolos_gate"] = summary_df["skill_score"] >= threshold
    summary_df = summary_df.sort_values("skill_score", ascending=False)

    out_path = results_dir / "evaluation" / "all_experiments_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"SELESAI: {n_ok} berhasil, {n_skip} dilewati (sudah ada), "
          f"{n_fail} gagal, dari {len(combos)} total kombinasi.")
    print(f"Ringkasan lengkap -> {out_path}")
    print(f"\nTop 10 kombinasi berdasarkan Skill Score:")
    cols = ["run_name", "rmse", "mae", "r2", "nse", "skill_score", "lolos_gate"]
    print(summary_df[cols].head(10).to_string(index=False))

    n_lolos = int(summary_df["lolos_gate"].sum())
    print(f"\n{n_lolos}/{len(summary_df)} kombinasi lolos "
          f"skill_score_threshold={threshold}")


if __name__ == "__main__":
    main()