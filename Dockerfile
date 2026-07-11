# =============================================================================
# Dockerfile — thesis-sst
# Pengembangan Model Hibrida LSTM-Transformer untuk Prediksi SST Laut Banda
# Ferry Yonathan — Universitas Pamulang S-2, 2026
# =============================================================================
#
# Build targets:
#   cpu  (default) → Mac Mini M2 Pro: preprocessing, EDA, logic testing
#   cuda           → HPC BMKG: Ubuntu 22.04 + CUDA 12.1 + GPU training
#
# ⚠️  PENTING — Apple Silicon (M2 Pro):
#   PyTorch MPS backend TIDAK tersedia di dalam Docker container karena
#   Apple Metal framework hanya bisa diakses langsung dari host macOS.
#
#   Gunakan Docker untuk  : preprocessing, EDA, uji logika kode, HPC deploy
#   Gunakan native .venv  : training penuh dengan MPS GPU di Mac Mini
#
# Build commands:
#   docker build --build-arg TARGET=cpu  -t thesis-sst:cpu  .
#   docker build --build-arg TARGET=cuda -t thesis-sst:cuda .
#
# Konversi ke Singularity untuk HPC BMKG:
#   ./docker/build_for_hpc.sh
# =============================================================================

ARG TARGET=cpu

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — cpu: Python 3.11 slim (Mac / CI / preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS stage-cpu

ENV TORCH_DEVICE=cpu \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
        libhdf5-dev libnetcdf-dev \
        libgeos-dev libproj-dev proj-data proj-bin \
        libgdal-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheel (jauh lebih kecil dari CUDA, ~250 MB vs ~2 GB)
RUN pip install --upgrade pip && \
    pip install torch==2.3.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — cuda: CUDA 12.1 + cuDNN 8 (HPC BMKG, Ubuntu 22.04)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS stage-cuda

ENV TORCH_DEVICE=cuda \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Python 3.11 tidak tersedia di Ubuntu 22.04 default repo, perlu deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip python3.11-distutils \
        git build-essential \
        libhdf5-dev libnetcdf-dev \
        libgeos-dev libproj-dev proj-data proj-bin \
        libgdal-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# pip untuk Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# PyTorch CUDA 12.1
RUN pip install --upgrade pip && \
    pip install torch==2.3.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

# ─────────────────────────────────────────────────────────────────────────────
# FINAL STAGE — pilih berdasarkan TARGET
# ─────────────────────────────────────────────────────────────────────────────
FROM stage-${TARGET} AS final

ARG TARGET=cpu
ARG BUILD_DATE
ARG GIT_COMMIT

WORKDIR /workspace

# ── Install dependencies project ────────────────────────────────────────────
COPY requirements.txt requirements-dev.txt ./

RUN pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# ── Buat semua direktori project terlebih dahulu ─────────────────────────────
# Dilakukan sebelum COPY agar direktori selalu ada meski repo belum lengkap
RUN mkdir -p \
    src configs notebooks scripts \
    data/raw/cmems_reanalysis \
    data/raw/cmems_analysis \
    data/raw/argo_float \
    data/processed/train \
    data/processed/validation \
    data/processed/test \
    models/checkpoints \
    models/results \
    reports/figures \
    reports/tables \
    logs

# ── Copy source code ─────────────────────────────────────────────────────────
# Hanya copy direktori yang ADA di repo saat build.
# Direktori kosong (notebooks, scripts) sudah dibuat oleh RUN di atas.
# Data, model weights, dan reports di-mount sebagai Docker volumes.
COPY src/        src/
COPY configs/    configs/
# notebooks/ dan scripts/ di-copy hanya jika sudah ada isinya;
# jika belum, direktori kosong dari RUN mkdir di atas sudah cukup.
# Keduanya juga di-mount sebagai volume di docker-compose.yml.
RUN if [ -d "notebooks" ] && [ "$(ls -A notebooks 2>/dev/null)" ]; then \
        echo "notebooks/ sudah ada, skip (di-mount sebagai volume)"; \
    fi

# ── Entrypoint ───────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Metadata image ───────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="thesis-sst" \
      org.opencontainers.image.description="Hybrid LSTM-Transformer SST Prediction — Laut Banda" \
      org.opencontainers.image.authors="Ferry Yonathan <241012000099@unpam.ac.id>" \
      org.opencontainers.image.version="1.0.0" \
      build.target="${TARGET}" \
      build.date="${BUILD_DATE}" \
      build.git_commit="${GIT_COMMIT}"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]