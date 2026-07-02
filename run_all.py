"""Orkestrator end-to-end pipeline tesis SST.

Tahapan (pilih via --steps):
  download   -> unduh CMEMS (perlu kredensial copernicusmarine)
  preprocess -> NetCDF -> deret waktu SST bersih
  experiment -> 64 run (4 lookback x 4 horizon x 4 model)
  analyze    -> figur + ringkasan

Contoh:
  python run_all.py --steps preprocess experiment analyze
  python run_all.py --steps all
"""
from __future__ import annotations
import argparse
import yaml

from src.data import download as dl
from src.data import preprocess as pp
from src.training import hyperparameter_search as hs
from src.evaluation import visualize as viz

STEPS = ["download", "preprocess", "experiment", "analyze"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--steps", nargs="+", default=["all"])
    a = p.parse_args()
    cfg = load_config(a.config)
    steps = STEPS if a.steps == ["all"] else a.steps

    data_csv = f"{cfg['data']['processed_dir']}/sst_series.csv"
    results_csv = f"{cfg['evaluation']['results_dir']}/results_all.csv"

    if "download" in steps:
        dl.download(cfg)
    if "preprocess" in steps:
        pp.main(cfg)
    if "experiment" in steps:
        hs.run(cfg, data_csv)
    if "analyze" in steps:
        viz.main(cfg, results_csv)


if __name__ == "__main__":
    main()
