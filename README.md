# Dokumentasi Project: Prediksi SST Laut Banda
## LSTM, Transformer, dan Hybrid LSTM-Transformer pada Virtual Mooring

> Dokumen ini merangkum spesifikasi desain eksperimen yang sudah
> digunakan, struktur direktori project, dan daftar file pemrosesan
> data — disusun untuk keperluan dokumentasi git repository.
>
> **Catatan cakupan:** dokumen ini mencakup seluruh komponen yang
> dibangun dan diverifikasi bersama pada sesi kerja ini. Beberapa file
> lain diketahui ada di repository (dari laporan `ls` pengguna) namun
> isinya belum pernah direview — ditandai eksplisit pada Bagian 5.

---

## 1. Ringkasan Project

| | |
|---|---|
| Domain penelitian | Laut Banda, Indonesia |
| Target prediksi | Sea Surface Temperature (SST) |
| Pendekatan | Virtual mooring (6 titik representatif) |
| Model | LSTM, Transformer, Hybrid LSTM-Transformer |
| Pendekatan data | Multivariat (F=9 fitur) |
| Periode data | 2014–2025 |

---

## 2. Spesifikasi Desain Eksperimen

### 2.1 Domain Spasial

```yaml
domain:
  lat_min: -9.0   lat_max: -3.0
  lon_min: 123.0  lon_max: 133.0
  buffer:  -9.5 s/d -2.5 (lat), 122.5 s/d 133.5 (lon)
  depth: permukaan (~0.5 m)
```

### 2.2 Enam Titik Virtual Mooring

| Titik | Lon | Lat | Representasi |
|---|---|---|---|
| Lok-1 | 123.82 | -7.27 | Barat daya — jalur inflow Selat Ombai/Laut Flores |
| Lok-2 | 125.27 | -4.97 | Barat laut — jalur inflow Laut Banda utara–Buru |
| Lok-3 | 128.49 | -7.76 | Selatan-tengah — deep basin, zona upwelling |
| Lok-4 | 128.75 | -5.14 | Tengah — pusat basin (open ocean) |
| Lok-5 | 130.84 | -4.65 | Timur laut — pengaruh perairan Seram |
| Lok-6 | 130.93 | -6.13 | Timur — zona upwelling monsun tenggara |

### 2.3 Sumber Data & Peran

| Sumber | Dataset ID | Variabel | Peran | Periode dipakai |
|---|---|---|---|---|
| CMEMS GLORYS | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | `thetao` | Target (y) — train & val | 2014–2023 |
| CMEMS ANFC | `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` | `thetao` | Target (y) — test | 2024–2025 (file 2026 dikecualikan) |
| ERA5 | single-levels reanalysis | `u10,v10,t2m,d2m,sp` | Prediktor (X) | 2014–2025, seluruh split |

Catatan: variabel radiasi ERA5 (`ssr`,`str`) dikeluarkan dari desain
(kendala akses data daily-aggregated); digantikan fitur temporal
`doy_sin`/`doy_cos`. Variabel `sst` versi ERA5 hanya dipakai sebagai
pembanding QC independen, tidak masuk sebagai fitur model.

### 2.4 Variabel yang Digunakan

#### 2.4.1 Variabel Input — X (fitur prediktor, F = 9)

Setiap sampel input berbentuk tensor **(T, F)** = (panjang lookback, 9
fitur), dinormalisasi z-score (fit dari periode train saja).

| # | Variabel | Sumber | Satuan | Kegunaan |
|---|---|---|---|---|
| 1 | `sst` | GLORYS/ANFC (`thetao`, lapisan permukaan ~0.5 m) | °C | Histori target itu sendiri (autoregresif) — sinyal terkuat krn SST sangat autokorelatif |
| 2 | `u10` | ERA5 | m/s | Komponen zonal angin 10 m — arah/kekuatan monsun, pemicu upwelling |
| 3 | `v10` | ERA5 | m/s | Komponen meridional angin 10 m |
| 4 | `t2m` | ERA5 | °C (dari K) | Suhu udara 2 m — proxy fluks panas sensibel |
| 5 | `d2m` | ERA5 | °C (dari K) | Titik embun 2 m — proxy kelembapan & fluks panas laten (evaporative cooling) |
| 6 | `sp` | ERA5 | Pa | Tekanan permukaan — penanda sistem sinoptik/monsun |
| 7 | `wind_speed` | Turunan: √(u10² + v10²) | m/s | Magnitudo angin — pengaduk mixed layer, pengendali evaporasi |
| 8 | `doy_sin` | Turunan: sin(2π·doy/365.25) | — | Encoding musiman siklik (pengganti implisit sinyal radiasi yg di-drop) |
| 9 | `doy_cos` | Turunan: cos(2π·doy/365.25) | — | Pasangan doy_sin — posisi kalender kontinu tanpa lompatan 31 Des→1 Jan |

#### 2.4.2 Variabel Target — y

- **y = `sst`** (GLORYS/ANFC) pada **h hari ke depan** setelah akhir
  window input. Bentuk **(H,)** per sampel — prediksi *direct
  multi-step* (semua langkah horizon sekaligus, bukan rekursif).
- **Tidak dinormalisasi** — tetap skala °C asli, sehingga RMSE/MAE
  langsung terbaca dalam satuan suhu.

#### 2.4.3 Variabel Metadata (bukan input model, untuk analisis)

| Variabel | Lokasi | Kegunaan |
|---|---|---|
| `mooring_id` (0–5) | `meta_*.csv` | Identitas titik — pemecahan evaluasi per lokasi; belum dipakai sbg fitur/embedding model |
| `target_start_date` | `meta_*.csv` | Tanggal awal periode target tiap window — analisis temporal & plotting |
| `sst_source` (GLORYS/ANFC) | `mooring_*.csv` | Penanda asal data — dasar analisis distribution shift |
| `sst_era5` | `mooring_*.csv` | SST versi ERA5/OSTIA — HANYA pembanding independen QC, sengaja TIDAK jadi fitur X (mencegah kebocoran informasi dari produk lain) |

#### 2.4.4 Struktur Tensor

```
X : (N, T, F) = (jumlah_window, lookback 7/14/21/30, 9 fitur)
y : (N, H)    = (jumlah_window, horizon 1/3/7/14)
```
Tidak ada dimensi spasial (grid lat/lon) — pendekatan virtual mooring
mereduksi masalah spatio-temporal jadi deret waktu titik, sehingga
input cukup 2D per sampel, bukan 4D seperti model berbasis grid
(ConvLSTM/ViT).

### 2.5 Skema Split (Kronologis, Bukan Acak)

| Split | Periode | Sumber |
|---|---|---|
| Train | 2014-01-01 s/d 2022-12-31 | GLORYS |
| Val | 2023-01-01 s/d 2023-12-31 | GLORYS |
| Test | 2024-01-01 s/d 2025-12-31 | ANFC (out-of-distribution) |

Normalisasi (z-score) di-fit **hanya dari periode train**, diterapkan
ke val/test tanpa penghitungan ulang (pencegahan data leakage). Target
tidak dinormalisasi (skala °C asli).

### 2.6 Grid Eksperimen Utama

```
lookback_windows: [7, 14, 21, 30] hari
horizons:         [1, 3, 7, 14] hari
models:           [lstm, transformer, hybrid]
```
= **4 × 4 × 3 = 48 eksperimen**

**Eksperimen tambahan (ad-hoc, di luar grid utama):** lookback=1 hari
untuk ketiga model × 4 horizon (12 kombinasi), dijalankan via CLI
override (`--lookbacks 1`) tanpa mengubah `config.yaml`, khusus untuk
replikasi format tabel penelitian acuan (lookback 1/7/14/21).

### 2.7 Arsitektur Model

| Model | Struktur | Parameter (approx.) |
|---|---|---|
| LSTM | LSTM(hidden=50, layer=1) → Linear | ~12 ribu |
| Transformer | Linear proj(d=64) → PosEnc → TransformerEncoder(2 layer, 4 head) → Linear | ~68 ribu |
| Hybrid | LSTM(hidden=50) → proj Linear(→52) → PosEnc → TransformerEncoder(2 layer, 4 head) → Linear | ~65 ribu |

### 2.8 Konfigurasi Training

```yaml
device: cuda
batch_size: 64
max_epochs: 100
learning_rate: 0.001
early_stopping_patience: 10
random_seed: 42
```

### 2.9 Metrik Evaluasi

**Dipakai dalam narasi thesis (Bab IV):** MAE, RMSE, R².

**Dihitung otomatis oleh `evaluate.py`/`run_all.py` tapi TIDAK dipakai
dalam narasi thesis:** NSE, Skill Score (terhadap baseline Persistence).
Keputusan ini diambil setelah evaluasi awal — narasi akhir bab hasil
berbasis RMSE/MAE/R² murni, bukan Skill Score.

### 2.10 Infrastruktur Eksekusi

- **HPC:** SLURM, partition `gpu_riset`, NVIDIA H100 80GB (`--gres=gpu:1`)
- **Container:** Apptainer (`thesis-sst.sif`), flag `--nv` wajib untuk akses GPU
- **Working directory:** `/workspace` di dalam container (bind mount dari `~/thesis-sst`)

---

## 3. Struktur Direktori Project

```
thesis-sst/
├── configs/
│   └── <a href="https://github.com/feyon/thesis-sst/blob/main/configs/config.yaml">config.yaml</a>                  # single source of truth seluruh pipeline
│
├── data/
│   ├── raw/
│   │   ├── cmems_reanalysis/        # glorys_thetao_2014..2023.nc
│   │   ├── cmems_analysis/          # anfc_thetao_2024..2026.nc (2026 tdk dipakai)
│   │   ├── dataset_laut_banda/
│   │   │   ├── era5_banda_2014..2025.nc      # raw (ternyata ZIP berekstensi .nc)
│   │   │   └── era5_banda_fixed/             # hasil ekstraksi fix_era5_zip.py
│   │
│   └── processed/
│       ├── virtual_mooring/
│       │   ├── mooring_01.csv .. mooring_06.csv
│       │   ├── mooring_grid_info.csv
│       │   └── qc/                                # qc_summary_all.csv, qc_*.png
│       └── windowed/
│           ├── scaler_params.csv
│           ├── lb07_h01/ .. lb30_h14/    # 16 kombinasi grid utama
│           └── lb01_h01/ .. lb01_h14/    # 4 kombinasi ad-hoc (lookback=1)
│               ├── X_train.npy, y_train.npy, meta_train.csv
│               ├── X_val.npy,   y_val.npy,   meta_val.csv
│               └── X_test.npy,  y_test.npy,  meta_test.csv
│
├── src/
│   ├── data_prep/          # lihat Bagian 4
│   ├── training/           # train.py
│   ├── evaluation/         # evaluate.py, plot_*.py, build_comparison_tables.py, dll
│   └── experiments/        # run_all.py
│
└── results/
    ├── checkpoints/<model>_lb{N}_h{H}/
    │   ├── best_model.pt, history.csv, summary.json
    ├── evaluation/
    │   ├── <model>_lb{N}_h{H}_<split>/
    │   │   ├── metrics_overall.csv, metrics_per_horizon.csv
    │   │   ├── metrics_per_mooring.csv, predictions.npz
    │   └── all_experiments_summary.csv
    ├── tables/              # comparison_<model>_h{H}_<split>.csv/.md
    ├── figures/             # fig_*.png per run + metric_lines_lb{N}_<split>/
    └── samples/             # test_sample_lb{N}_h{H}[_model].csv
```

---

## 4. File Pemrosesan Data (`src/data_prep/`)

Urutan eksekusi pipeline utama (yang dipakai dalam thesis final):

| # | File | Fungsi |
|---|---|---|
| 1 | `fix_era5_zip.py` | Memperbaiki file ERA5 yang ternyata ZIP archive berekstensi `.nc`; ekstrak ke `era5_banda_fixed/` |
| 2 | `diagnose_era5_files.py` | Diagnostik awal — cek file ERA5 mana yang gagal dibuka xarray sebelum perbaikan |
| 3 | `extract_virtual_mooring.py` | **Script utama ekstraksi.** Menarik SST (GLORYS+ANFC) dan 5 prediktor ERA5 ke 6 titik mooring, menghasilkan `mooring_01..06.csv` |
| 4 | `qc_virtual_mooring.py` | Quality control: ringkasan statistik, plot deret waktu SST, klimatologi musiman, distribusi prediktor |
| 5 | `build_windows.py` | Sliding-window + split kronologis untuk seluruh grid eksperimen; mendukung override CLI `--lookbacks`/`--horizons` untuk eksperimen ad-hoc |

File pendukung/eksploratif (dipakai pada tahap awal, sebagian hasilnya
digantikan oleh script di atas):

| File | Fungsi |
|---|---|
| `extract_sst_only.py` | Versi awal ekstraksi SST-only (GLORYS+ANFC), sebelum ERA5 lengkap tersedia |

File terkait validasi Sentinel-3 — **dibangun dan diuji, namun
validasi satelit akhirnya dikeluarkan dari cakupan thesis final**
(keterbatasan cakupan data, keputusan eksplisit pengguna):

| File | Fungsi |
|---|---|
| `inspect_sentinel_sample.py` | Inspeksi struktur 1 sampel file Sentinel-3 sebelum ekstraksi massal |
| `diagnose_sentinel_zips.py` | Diagnostik integritas ZIP Sentinel-3 (cek EOCD, deteksi file terpotong) |
| `extract_sentinel_mooring.py` | Ekstraksi Sentinel-3 dari ZIP saja (versi awal, sebelum digabung dgn folder `satelit/`) |
| `extract_sentinel_combined.py` | Ekstraksi gabungan dua sumber (ZIP valid + folder `satelit/` sbg pelengkap gap tanggal) |
| `analyze_sentinel_validation.py` | Analisis bias ANFC vs Sentinel-3 dan uji jembatan distribution shift GLORYS↔ANFC |

---

## 5. File Training, Evaluasi, dan Eksperimen

### 5.1 `src/training/`

| File | Fungsi |
|---|---|
| `train.py` | Training satu kombinasi (lookback, horizon, model) dari CLI; mendukung ketiga arsitektur via `--model`; early stopping berbasis val RMSE |

### 5.2 `src/evaluation/` — dibangun dalam sesi ini

| File | Fungsi |
|---|---|
| `evaluate.py` | Evaluasi 1 kombinasi vs baseline Persistence; hasilkan metrik overall/per-horizon/per-mooring + `predictions.npz` |
| `plot_results.py` | 6 jenis gambar hasil (learning curve, prediksi vs observasi, degradasi horizon, per mooring, heatmap grid); opsi `--no_persistence` |
| `plot_metric_lines.py` | Grafik garis MAE/RMSE/R² vs Rentang Prediksi, dipecah per titik mooring, per model |
| `build_comparison_tables.py` | Tabel format P1–P6 × lookback (gaya penelitian acuan); mendukung override `--lookbacks` |
| `export_test_sample.py` | Ekspor sampel test set (fitur asli + target + prediksi) dalam tabel mudah dibaca |

### 5.3 `src/experiments/`

| File | Fungsi |
|---|---|
| `run_all.py` | Orkestrasi seluruh grid (loop train+evaluate per kombinasi), agregasi `all_experiments_summary.csv`; mendukung `--skip_existing`, `--lookbacks`, `--horizons`, `--models` |

### 5.4 File Lain di `src/evaluation/` — **Isi Belum Direview**

File berikut diketahui ada di repository (dari daftar direktori yang
dilaporkan pengguna) namun **belum pernah ditinjau isinya** pada sesi
kerja ini. Deskripsi fungsi di bawah ini bersifat dugaan berdasarkan
nama file saja — perlu diverifikasi langsung ke isi file sebelum
didokumentasikan lebih lanjut:

- `collocation_argo.py` — diduga terkait kolokasi data Argo float
- `validate_argo.py` — diduga terkait validasi Argo float
- `visualize_argo.py` — diduga visualisasi data Argo
- `plot_argo_distribution.py` — diduga plot distribusi spasial Argo
- `evaluate_along_track.py` — diduga evaluasi along-track (mis. thd satelit)
- `evaluate_along_track_ablation.py` — diduga versi ablasi dari file di atas
- `recompute_metrics.py` — diduga penghitungan ulang metrik
- `metrics.py` — diduga modul fungsi metrik terpisah
- `visualize_bab4.py` — diduga skrip visualisasi khusus Bab IV
- `visualize.py` — diduga modul visualisasi umum

File lain yang disebutkan namun belum diketahui isinya:
`src/training/train_lstm_baseline.py`.

**Catatan penting:** file-file pada bagian ini mengindikasikan mungkin
ada pekerjaan terkait validasi Argo float yang sudah dimulai di luar
sesi kerja ini. Perlu klarifikasi status pekerjaan tersebut agar
dokumentasi project tetap akurat dan tidak terjadi duplikasi kerja.

---

## 6. Riwayat Keputusan Desain Penting

| Keputusan | Alasan |
|---|---|
| Pendekatan univariate → multivariate | Data ERA5 lengkap tersedia setelah desain awal disusun |
| Radiasi ERA5 di-drop, diganti `doy_sin`/`doy_cos` | Kendala akses data daily-aggregated |
| Ekstraksi nearest-neighbor, bukan interpolasi spasial | Merepresentasikan kondisi grid cell aktual (virtual mooring) |
| Split kronologis, bukan `train_test_split` acak | Mencegah temporal leakage pada data deret waktu |
| Scaler di-fit hanya dari train | Mencegah kebocoran statistik dari data val/test |
| Test set = ANFC (beda produk dari train GLORYS) | Uji generalisasi out-of-distribution yang sengaja, bukan split acak dari populasi homogen |
| Validasi Sentinel-3 dikeluarkan dari scope final | Cakupan data rusak/tidak lengkap (29/43 ZIP gagal, 26 bulan kosong) |
| Skill Score dihitung tapi tidak dipakai di narasi thesis | Keputusan eksplisit — fokus narasi pada MAE/RMSE/R² |
| Eksperimen ad-hoc lookback=1 di luar grid utama | Replikasi format tabel penelitian acuan tanpa mengubah desain 48-eksperimen resmi |
