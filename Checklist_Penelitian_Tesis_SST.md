# Checklist Penelitian Tesis — Prediksi SST Laut Banda

> Model Hibrida LSTM-Transformer · Ferry Yonathan (NIM 241012000099)
> Dokumen ini disusun berdasarkan urutan langkah kerja, tanpa target tanggal.
> Isi kolom log di tiap fase saat langkah selesai dikerjakan, untuk dijadikan bukti/lampiran metodologi tesis.

**Cara pakai:** Centang `[x]` tiap butir yang selesai. Isi blok "Log Fase" di akhir tiap fase sebagai catatan aktivitas — ini bisa langsung jadi bahan lampiran logbook penelitian.

---

## Fase 0 — Perbaikan Teknis Awal

- [ ] Update `analysis_dataset_id` di `configs/config.yaml` → `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m`
- [ ] Koreksi `reanalysis_end` dari `"2023-12-31"` → `"2022-12-31"` di `configs/config.yaml`
- [ ] Ubah `raw.glob("*.nc")` → `raw.rglob("*.nc")` di `preprocess.py` agar scan `data/raw/cmems_reanalysis/` dan `data/raw/cmems_analysis/`
- [ ] Commit & push perubahan ke repo `ferryyon1102/thesis-sst`
- [ ] Verifikasi ulang Docker image `thesis-sst:cpu` masih build sukses setelah perubahan

**Log Fase 0**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti (commit hash, screenshot, dsb.): _______________________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 1 — Preprocessing Data

- [ ] Jalankan pipeline preprocessing (`python -m scripts.run_pipeline`) di MacBook Air (Docker)
- [ ] QC: cek shape output (grid 85×133), tipe float32, tidak ada NaN yang tidak wajar
- [ ] QC: pastikan normalisasi z-score fit hanya pada train set (2014–2022), verifikasi train/val/test terpisah
- [ ] Simpan hasil ke `data/processed/{train,validation,test}`
- [ ] Catat ringkasan statistik data (mean, std, range) sebagai bahan lampiran BAB III

**Log Fase 1**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti: _______________________________________________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 2 — Pembersihan Dokumen Tesis (paralel, tidak menghambat fase teknis)

- [ ] Revisi `BAB_III_3_1_Analisa_Kebutuhan.docx.md`: hapus referensi 5 variabel CMEMS multivariat + ERA5
- [ ] Hapus footnote atmosferik pada Tabel 3.1
- [ ] Hapus baris "Tabel 3.2 Spesifikasi Data Atmospheric ERA5" dari Daftar Tabel/Daftar Isi
- [ ] Ganti sitasi Xu, Q., Chen, P., Mao, Y., & Zhong, Y. (2023) dengan Murphy (1988) untuk konsep Skill Score
- [ ] Tambahkan framing eksplisit: ambang SS > 0.3 adalah keputusan operasional penelitian, bukan baku dari literatur
- [ ] Perkuat justifikasi desain univariate (SST-only) sebagai delimitasi yang disengaja, bukan keterbatasan
- [ ] Review konsistensi terminologi di seluruh BAB I–III (pastikan tidak ada sisa referensi multivariat lain)
- [ ] Kirim BAB I–III versi bersih ke pembimbing untuk direview

**Log Fase 2**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti: _______________________________________________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 3 — HPC Canary Test (opsional, sebelum commit penuh ke HPC)

- [ ] Convert image `thesis-sst:cpu` (Docker) → format Apptainer/Singularity `.sif`
- [ ] Siapkan/verifikasi SLURM job script untuk 1 job percobaan (1 model, 1 kombinasi lookback/horizon, 1–2 epoch)
- [ ] Submit job percobaan pertama, catat waktu tunggu antrean + waktu eksekusi
- [ ] Ulangi submit job percobaan pada waktu berbeda (melihat variasi antrean shared queue)
- [ ] Bandingkan waktu eksekusi HPC vs Mac Mini (MPS) untuk beban kerja yang sama
- [ ] Putuskan strategi final: HPC penuh / hybrid (HPC untuk model berat, Mac Mini untuk model ringan) / tetap Mac Mini saja
- [ ] Dokumentasikan keputusan & alasannya (bahan bagian infrastruktur/metodologi tesis)

**Log Fase 3**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti (waktu antre, waktu eksekusi, keputusan akhir): __________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 4 — Hyperparameter Tuning

- [ ] Bangun dataset sliding window untuk seluruh kombinasi lookback (7/14/21/30) × horizon (1/3/7/14)
- [ ] Tentukan representative subset untuk tuning (1–2 kombinasi lookback/horizon per arsitektur)
- [ ] Jalankan hyperparameter search (learning rate, hidden size, jumlah layer/head, dropout, batch size) pada data validasi 2023, per arsitektur (LSTM, Transformer, Hybrid)
- [ ] Catat hasil tuning & alasan pemilihan hyperparameter final
- [ ] Simpan hyperparameter final ke `configs/config.yaml` per model
- [ ] Dokumentasikan proses tuning untuk BAB III/BAB IV (tabel hyperparameter final)

**Log Fase 4**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti (tabel hyperparameter final): ___________________________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 5 — Training 48 Eksperimen

- [ ] Siapkan script batch runner untuk 48 kombinasi (3 model × 4 lookback × 4 horizon), sesuai strategi Fase 3
- [ ] Jalankan training, prioritaskan horizon pendek (h=1,3) dahulu
- [ ] Pantau proses training berkala (loss curve, konvergensi, indikasi overfitting)
- [ ] Simpan checkpoint & log training tiap eksperimen ke `models/checkpoints/`
- [ ] Verifikasi seluruh 48 eksperimen selesai tanpa error/crash
- [ ] Catat waktu training per eksperimen (bahan bagian computational cost di tesis)

**Log Fase 5**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti (jumlah eksperimen selesai, lokasi checkpoint): _________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 6 — Evaluasi Test Set

- [ ] Hitung RMSE, MAE, MAPE, R², NSE untuk seluruh 48 eksperimen pada test set (2024–2026)
- [ ] Hitung Skill Score terhadap Persistence sebagai baseline silent (ambang referensi SS > 0.3)
- [ ] Susun tabel perbandingan hasil (per model × lookback × horizon)
- [ ] Identifikasi model & konfigurasi terbaik per horizon
- [ ] Buat visualisasi degradasi akurasi terhadap horizon prediksi
- [ ] Simpan hasil ke `models/results/` dan `reports/tables/`

**Log Fase 6**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti (lokasi tabel/figure): _________________________________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 7 — Validasi Independen Data Float BMKG

- [ ] Load data Argo float BMKG (2022–2026)
- [ ] Lakukan spatial collocation dengan KDTree nearest-neighbor terhadap grid CMEMS
- [ ] Hitung bias rata-rata sistematis, RMSE eksternal, korelasi Pearson (prediksi vs observasi float)
- [ ] Bandingkan hasil validasi float dengan hasil evaluasi test set (cek konsistensi)
- [ ] Buat visualisasi hasil validasi (scatter plot, time series overlay)
- [ ] Simpan hasil ke `reports/figures/` dan `reports/tables/`

**Log Fase 7**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti: _______________________________________________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 8 — Penulisan BAB IV

- [ ] Susun outline BAB IV berdasarkan tabel & figure yang tersedia
- [ ] Tulis analisis variabilitas spasio-temporal SST (data reanalisis)
- [ ] Tulis hasil pelatihan & evaluasi model di berbagai horizon
- [ ] Tulis perbandingan performa LSTM vs Transformer vs Hybrid
- [ ] Tulis analisis degradasi akurasi antar horizon
- [ ] Tulis hasil validasi data float
- [ ] Tulis analisis spasio-temporal hasil prediksi
- [ ] Review internal: pastikan seluruh rumusan masalah (butir 1.2.3 proposal) terjawab

**Log Fase 8**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti (draft BAB IV): _________________________________________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 9 — Penulisan BAB V

- [ ] Tulis kesimpulan yang menjawab tiap rumusan masalah & tujuan penelitian (butir 1.3 proposal)
- [ ] Tulis keterbatasan penelitian (termasuk justifikasi desain univariate sebagai delimitasi)
- [ ] Tulis saran untuk penelitian selanjutnya
- [ ] Finalisasi Daftar Pustaka — verifikasi ulang seluruh sitasi (tidak ada yang tidak terverifikasi)

**Log Fase 9**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti (draft BAB V): _________________________________________________________
- Kendala yang ditemui: ________________________________________________________

---

## Fase 10 — QA & Finalisasi Dokumen

- [ ] Cek format keseluruhan: Times New Roman 12pt, spasi 1.5, rata kanan-kiri, margin 4-3-3-3 cm
- [ ] Cek konsistensi referensi APA 7th edition di seluruh dokumen
- [ ] Update Daftar Isi, Daftar Tabel, Daftar Gambar sesuai isi final
- [ ] Konversi dokumen: LibreOffice/soffice → PDF
- [ ] Render PDF ke gambar (pdftoppm) untuk QA visual halaman per halaman
- [ ] Kirim draft lengkap ke pembimbing untuk review akhir
- [ ] Revisi berdasarkan feedback pembimbing
- [ ] Finalisasi & submit tesis

**Log Fase 10**
- Tanggal mulai / selesai: _______________ / _______________
- Output/bukti (versi final, tanggal submit): _________________________________________
- Kendala yang ditemui: ________________________________________________________

---

## Ringkasan Status (isi ulang berkala)

| Fase | Status | Tanggal Selesai |
|---|---|---|
| 0. Perbaikan Teknis Awal | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 1. Preprocessing Data | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 2. Pembersihan Dokumen | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 3. HPC Canary Test | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 4. Hyperparameter Tuning | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 5. Training 48 Eksperimen | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 6. Evaluasi Test Set | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 7. Validasi Float BMKG | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 8. Penulisan BAB IV | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 9. Penulisan BAB V | ☐ Belum / ☐ Proses / ☐ Selesai | |
| 10. QA & Finalisasi | ☐ Belum / ☐ Proses / ☐ Selesai | |
