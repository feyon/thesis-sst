# =============================================================================
# Dockerfile — thesis-sst
# Build targets:
#   cpu  (default) -> Mac Mini M2 Pro: preprocessing, EDA, testing
#   cuda           -> HPC BMKG: Ubuntu 22.04 + CUDA 12.1
#
# MPS backend TIDAK tersedia di dalam Docker container.
# Gunakan native .venv untuk training dengan MPS di Mac Mini.
# =============================================================================

ARG TARGET=cpu

# -----------------------------------------------------------------------------
# STAGE 1 — cpu
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS stage-cpu

ENV TORCH_DEVICE=cpu \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
        libhdf5-dev libnetcdf-dev \
        libgeos-dev libproj-dev proj-data proj-bin \
        libgdal-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install torch==2.3.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# -----------------------------------------------------------------------------
# STAGE 2 — cuda (HPC BMKG)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS stage-cuda

ENV TORCH_DEVICE=cuda \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive

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

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN pip install --upgrade pip \
    && pip install torch==2.3.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------------------
# FINAL STAGE
# -----------------------------------------------------------------------------
FROM stage-${TARGET} AS final

ARG TARGET=cpu

WORKDIR /workspace

COPY requirements.txt requirements-dev.txt ./

RUN pip install -r requirements.txt \
    && pip install -r requirements-dev.txt

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

COPY src/     src/
COPY configs/ configs/

LABEL maintainer="Ferry Yonathan <241012000099@unpam.ac.id>" \
      description="Hybrid LSTM-Transformer SST Prediction Laut Banda" \
      build.target="${TARGET}"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
