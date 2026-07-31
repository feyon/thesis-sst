# Ringkasan Keluaran Project
## Prediksi SST Laut Banda dengan LSTM, Transformer, dan Hybrid LSTM-Transformer

---

## 1. Ringkasan Eksekutif

Project ini menghasilkan pipeline penelitian lengkap — dari akuisisi
data (CMEMS GLORYS/ANFC, ERA5), ekstraksi 6 titik virtual mooring,
hingga pelatihan dan evaluasi **60 model deep learning** (48 kombinasi
grid utama + 12 kombinasi tambahan) untuk prediksi SST di Laut Banda.
Seluruh pipeline berjalan di HPC (SLURM + Apptainer + GPU H100) dan
sepenuhnya reproducible lewat `config.yaml` sebagai single source of
truth.

---

## 2. Keluaran Data

| Keluaran | Jumlah/Cakupan | Lokasi |
|---|---|---|
| Data mentah terverifikasi | GLORYS 2014–2023, ANFC 2024–2025, ERA5 2014–2025 | `data/raw/` |
| Deret waktu virtual mooring | 6 file × 4.383 hari, F=9 fitur, 0 NaN | `data/processed/virtual_mooring/mooring_01..06.csv` |
| Dataset siap-latih (windowed) | 16 kombinasi grid utama + 4 kombinasi ad-hoc (lookback=1) | `data/processed/windowed/lb{N}_h{H}/` |
| Parameter normalisasi | 1 file (fit dari train saja) | `windowed/scaler_params.csv` |
| Laporan QC data | Statistik, klimatologi, cross-validation GLORYS↔ERA5 | `virtual_mooring/qc/` |

---

## 3. Keluaran Model

**60 model terlatih** (checkpoint + riwayat training + ringkasan per run):

| Kategori | Jumlah kombinasi |
|---|---|
| Grid utama (4 lookback × 4 horizon × 3 model) | 48 |
| Ad-hoc lookback=1 (3 model × 4 horizon) | 12 |
| **Total** | **60** |

Tiap kombinasi menghasilkan di `results/checkpoints/<model>_lb{N}_h{H}/`:
- `best_model.pt` — bobot model pada epoch terbaik (early stopping)
- `history.csv` — RMSE/MAE train & val per epoch
- `summary.json` — jumlah parameter, waktu training, epoch terbaik

---

## 4. Keluaran Evaluasi

| Keluaran | Deskripsi | Lokasi |
|---|---|---|
| Ringkasan grid 48 eksperimen | RMSE, MAE, R² (+ NSE, Skill Score dihitung tapi tdk dipakai narasi) | `results/evaluation/all_experiments_summary.csv` |
| Metrik per kombinasi | Overall, per-horizon, per-mooring, prediksi mentah | `results/evaluation/<run>_<split>/` |
| Tabel format penelitian acuan | 3 tabel (P1–P6 × lookback 1/7/14/21), 1 per model | `results/tables/comparison_<model>_h01_test.{csv,md}` |
| Grafik hasil | Learning curve, prediksi vs observasi, degradasi horizon, per-mooring, heatmap grid | `results/figures/<run>_<split>/` |
| Grafik tren metrik | 9 grafik (MAE/RMSE/R² × 3 model) vs Rentang Prediksi | `results/figures/metric_lines_lb07_test/` |
| Sample data test set | Fitur asli + target + prediksi, format tabel | `results/samples/test_sample_*.csv` |

---

## 5. Temuan Utama

Ringkasan hasil analisis terhadap 48 eksperimen grid utama (split test,
data ANFC 2024–2025, split kronologis):

1. **Akurasi keseluruhan.** RMSE berkisar 0,1455–0,4045 °C (rata-rata
   0,2733 °C); R² berkisar 0,883–0,984 (rata-rata 0,9404) di seluruh
   grid.

2. **Model terbaik: LSTM.** Rata-rata RMSE terendah (0,2674 °C),
   diikuti Hybrid (0,2700 °C) dan Transformer (0,2826 °C). **Hybrid
   LSTM-Transformer tidak mengungguli LSTM murni** meski parameter
   ~5× lebih banyak.

3. **Rentang prediksi adalah faktor paling dominan.** Selisih RMSE
   rata-rata 0,2232 °C dari horizon 1 hari (0,1609 °C) ke 14 hari
   (0,3840 °C) — jauh melampaui pengaruh model (maks. 0,0151 °C) atau
   lookback (maks. 0,0047 °C).

4. **Transformer secara spesifik lemah di horizon pendek.** RMSE ~30%
   lebih tinggi dari LSTM pada horizon 1 hari, namun kompetitif pada
   horizon 14 hari — mengindikasikan mekanisme *self-attention* kurang
   efektif menangkap ketergantungan sangat jangka pendek.

5. **Variasi antar lokasi mooring melebihi variasi antar model.**
   Selisih RMSE Lok-1 (terburuk) vs Lok-4 (terbaik) mencapai 0,0975 °C
   (~30% relatif) — **berlawanan dari hipotesis awal**: Lok-1 yang
   paling stabil secara klimatologis justru paling sulit diprediksi,
   bukan zona upwelling (Lok-3/Lok-6) seperti dugaan semula.

6. **Peringkat faktor pengaruh akurasi (dari terbesar):** rentang
   prediksi > lokasi mooring > arsitektur model > panjang lookback.

---

## 6. Keluaran Dokumen (Bahan Penulisan Thesis)

| Dokumen | Isi |
|---|---|
| `ringkasan_pipeline_thesis_sst.md` | Narasi lengkap pipeline: pengumpulan data s/d hasil |
| `PROJECT_DOCUMENTATION.md` | Spesifikasi desain eksperimen, struktur direktori, variabel, daftar file |
| `kerangka_bab4.md` | Kerangka besar Bab IV — judul tiap subbab + rasional keberadaannya |
| `bab4_hasil_pembahasan_draft.md` | Draf lengkap Bab IV siap edit (9 subbab, seluruh data nyata tervalidasi) |
| Dokumen ini | Ringkasan keluaran project secara keseluruhan |

---

## 7. Cakupan yang Sengaja Dikeluarkan / Belum Selesai

| Item | Status |
|---|---|
| Validasi independen Sentinel-3 | Dibangun & diuji, dikeluarkan dari scope thesis final (cakupan data rusak/tidak lengkap) |
| Validasi independen Argo float | Belum dimulai dalam sesi kerja ini; kemungkinan ada progres terpisah (`collocation_argo.py` dkk. di repo, isi belum direview — lihat `PROJECT_DOCUMENTATION.md` §5.4) |
| Skill Score / NSE | Dihitung otomatis, tidak dipakai dalam narasi thesis (keputusan eksplisit) |
| MAPE | Dihapus dari pipeline evaluasi, hanya MAE/RMSE/R² yang dipakai |

---

*Dokumen ini merangkum keluaran per tahap Juli 2026 (mengikuti sesi
kerja terakhir). Angka pada Bagian 5 bersumber dari
`all_experiments_summary.csv` hasil run nyata di HPC, tervalidasi
silang dengan minimal satu sumber independen untuk tiap klaim utama.*
