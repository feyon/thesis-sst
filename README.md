# thesis-sst — Prediksi SST Laut Banda (Hibrida LSTM-Transformer)

Kerangka penelitian untuk menguji **kemampuan arsitektur hibrida LSTM-Transformer**
dalam memprediksi *Sea Surface Temperature* (SST) di Laut Banda.

> **Framing tesis (Ilmu Komputer):** fokus pada kapabilitas arsitektur, bukan
> analisis klimatologi. Pendekatan **univariat** — hanya `thetao` (SST) sebagai
> input — agar setiap keunggulan performa murni dapat diatribusikan ke arsitektur.

---

## Desain Eksperimen

| Dimensi | Nilai |
|---|---|
| Input | **Univariat**: `thetao` (SST), F = 1 |
| Lookback window (N) | 7, 14, 21, 30 hari |
| Horizon prediksi (h) | 1, 3, 7, 14 hari |
| Model dibandingkan | LSTM, Transformer, Hybrid |
| Baseline (acuan Skill Score) | Persistence — bukan model pembanding |
| Total run | 4 × 4 × 3 = **48 eksperimen** |
| Gate utama | **Skill Score > 0.3** vs Persistence (Xu et al., 2023) |
| Metrik | RMSE, MAE, R², NSE, Skill Score |

**Pembagian data (kronologis):** Train 2014–2022 · Val 2023 · Test 2024–2026.
**Domain:** Laut Banda 3°S–9°S / 123°E–133°E (buffer download lebih luas).

---

## Struktur Proyek

```
thesis-sst/
├── configs/config.yaml          # single source of truth (domain, split, hyperparam)
├── requirements.txt
├── run_all.py                   # orkestrator end-to-end
├── smoke_test.py                # validasi pipeline tanpa data (data sintetis)
└── src/
    ├── data/
    │   ├── download.py          # unduh CMEMS (copernicusmarine, per tahun)
    │   ├── preprocess.py        # QC + reduksi domain + interpolasi -> CSV
    │   └── dataset.py           # z-score (fit train-only) + sliding window
    ├── models/
    │   ├── lstm.py              # LSTM standalone
    │   ├── transformer.py       # Transformer encoder standalone
    │   ├── hybrid.py            # Hibrida sekuensial (proyeksi 50->52)
    │   ├── persistence.py       # baseline naif
    │   └── factory.py
    ├── training/
    │   ├── trainer.py           # Adam + early stopping (MPS/CPU/CUDA)
    │   └── hyperparameter_search.py   # 64 run
    └── evaluation/
        ├── metrics.py           # RMSE/MAE/R²/NSE/Skill Score
        └── visualize.py         # figur + ringkasan
```

---

## Instalasi

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# kredensial CMEMS (sekali saja):
copernicusmarine login
```

## Penggunaan

```bash
# 0. Validasi pipeline (tanpa data, ~10 detik)
python smoke_test.py

# 1. Unduh data CMEMS (univariat: hanya thetao)
python -m src.data.download --config configs/config.yaml

# 2. Pra-pemrosesan -> data/processed/sst_series.csv
python -m src.data.preprocess --config configs/config.yaml

# 3. Jalankan 64 eksperimen (semalam di Mac Mini M2 Pro)
python -m src.training.hyperparameter_search --config configs/config.yaml

# 4. Analisis + figur
python -m src.evaluation.visualize --config configs/config.yaml

# atau sekaligus:
python run_all.py --steps preprocess experiment analyze
```

---

## Catatan Metodologis

- **Anti-leakage:** statistik z-score di-*fit* **hanya pada set pelatihan**, lalu
  diterapkan ke val/test. Window dibentuk di dalam tiap split.
- **Skill Score** = `1 − MSE_model / MSE_persistence`. Karena input univariat,
  perbandingan dengan Persistence menjadi *apple-to-apple* (informasi sama).
- **Hybrid 50→52:** `lstm_hidden=50` tidak habis dibagi `nhead=4`; diselesaikan
  via lapisan proyeksi linear ke 52 sebelum Transformer.
- **MPS:** `pin_memory=false` wajib untuk backend Apple Silicon.

### Ekstensi ke validasi per-titik / Argo
`preprocess.py` saat ini mereduksi domain menjadi **satu deret rata-rata** (cukup
untuk menjalankan pipeline end-to-end). Untuk validasi Argo per-titik, ganti
langkah reduksi spasial dengan ekstraksi SST pada titik grid terkolokasi (KDTree),
menghasilkan array `[time, n_points]`, lalu jalankan windowing per titik.
