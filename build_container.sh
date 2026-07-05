#!/bin/bash
# ===========================================================================
# Script pembuatan container Singularity
# Jalankan di LOGIN NODE HPC BMKG (bukan compute node)
# ===========================================================================

DEF_FILE="thesis-sst.def"
SIF_FILE="thesis-sst.sif"

echo "============================================"
echo "Build Singularity Container — thesis-sst"
echo "Start: $(date)"
echo "============================================"

# Cek apakah Singularity/Apptainer tersedia
if command -v singularity &> /dev/null; then
    CMD="singularity"
    echo "Menggunakan: Singularity $(singularity --version)"
elif command -v apptainer &> /dev/null; then
    CMD="apptainer"
    echo "Menggunakan: Apptainer $(apptainer --version)"
else
    echo "ERROR: Singularity/Apptainer tidak ditemukan."
    echo "Hubungi admin HPC BMKG untuk mengaktifkan modul Singularity."
    exit 1
fi

# Build container
# --fakeroot: izin build tanpa root (tersedia di kebanyakan HPC modern)
# --force   : timpa .sif lama jika ada
$CMD build --fakeroot --force $SIF_FILE $DEF_FILE

echo "============================================"
echo "Build selesai: $(date)"
echo "File container: $SIF_FILE"
echo "Ukuran: $(du -sh $SIF_FILE | cut -f1)"
echo "============================================"

# Verifikasi container
echo "Verifikasi container..."
$CMD exec $SIF_FILE python -c "
import torch, xarray, numpy, pandas
print('Python  :', __import__('sys').version[:6])
print('PyTorch :', torch.__version__)
print('xarray  :', xarray.__version__)
print('numpy   :', numpy.__version__)
print('pandas  :', pandas.__version__)
print('CUDA    :', torch.cuda.is_available())
print('Container OK')
"
