#!/bin/bash
# =============================================================================
# docker/build_for_hpc.sh
# Build image CUDA dan konversi ke Singularity .sif untuk HPC BMKG
#
# Cara penggunaan:
#   chmod +x docker/build_for_hpc.sh
#   ./docker/build_for_hpc.sh
#
# Prasyarat:
#   - Docker Desktop terinstall dan berjalan
#   - Minimal ~15 GB ruang disk bebas (image CUDA besar)
#   - Singularity/Apptainer terinstall di HPC (sudah tersedia di HPC BMKG)
#
# Setelah selesai, upload .sif ke HPC BMKG:
#   scp thesis-sst.sif <username>@hpc.bmkg.go.id:~/thesis-sst/
# =============================================================================

set -euo pipefail

# ── Konfigurasi ───────────────────────────────────────────────────────────────
IMAGE_NAME="thesis-sst"
CUDA_TAG="${IMAGE_NAME}:cuda"
OUTPUT_TAR="${IMAGE_NAME}-cuda.tar.gz"
OUTPUT_SIF="${IMAGE_NAME}.sif"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     thesis-sst — Build untuk HPC BMKG (CUDA 12.1)       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Image    : ${CUDA_TAG}"
echo "  Output   : ${OUTPUT_TAR} → ${OUTPUT_SIF}"
echo "  Commit   : ${GIT_COMMIT}"
echo "  Date     : ${BUILD_DATE}"
echo ""

# ── Step 1: Build Docker image CUDA ──────────────────────────────────────────
echo "▶  [1/3] Build Docker image CUDA..."
echo "   Platform: linux/amd64 (diperlukan untuk kompatibilitas HPC x86_64)"
echo ""

docker build \
    --build-arg TARGET=cuda \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg GIT_COMMIT="${GIT_COMMIT}" \
    --platform linux/amd64 \
    --tag "${CUDA_TAG}" \
    --progress=plain \
    .

echo ""
echo "✓ Docker image berhasil dibuild: ${CUDA_TAG}"
echo "  Size: $(docker image inspect ${CUDA_TAG} --format='{{.Size}}' | awk '{printf "%.1f GB", $1/1024/1024/1024}')"

# ── Step 2: Export ke tar.gz ──────────────────────────────────────────────────
echo ""
echo "▶  [2/3] Export image ke ${OUTPUT_TAR}..."
echo "   (Proses ini mungkin memakan waktu 5-10 menit...)"

docker save "${CUDA_TAG}" | gzip > "${OUTPUT_TAR}"

TAR_SIZE=$(du -sh "${OUTPUT_TAR}" | cut -f1)
echo "✓ Export selesai: ${OUTPUT_TAR} (${TAR_SIZE})"

# ── Step 3: Instruksi upload & konversi Singularity ──────────────────────────
echo ""
echo "▶  [3/3] Langkah selanjutnya untuk HPC BMKG:"
echo ""
echo "   # 1. Upload ke HPC BMKG (dari Mac Mini)"
echo "   scp ${OUTPUT_TAR} <username>@hpc.bmkg.go.id:~/thesis-sst/"
echo ""
echo "   # 2. Login ke HPC BMKG"
echo "   ssh <username>@hpc.bmkg.go.id"
echo ""
echo "   # 3. Konversi ke Singularity .sif (di HPC)"
echo "   cd ~/thesis-sst"
echo "   singularity build ${OUTPUT_SIF} docker-archive://${OUTPUT_TAR}"
echo ""
echo "   # 4. Jalankan training dengan GPU di HPC (via Slurm)"
echo "   sbatch docker/slurm_job.sh"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "✓ Build selesai! File siap diupload ke HPC BMKG."
echo ""
