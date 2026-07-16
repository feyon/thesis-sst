#!/bin/bash
# =============================================================================
# docker/entrypoint.sh — thesis-sst container entrypoint
# =============================================================================
set -euo pipefail

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     thesis-sst: Hybrid LSTM-Transformer SST Prediction      ║"
echo "║     Prediksi Suhu Permukaan Laut — Perairan Laut Banda      ║"
echo "║     Ferry Yonathan — Universitas Pamulang S-2, 2026         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Info environment ─────────────────────────────────────────────────────────
PYTHON_VER=$(python --version 2>&1)
PYTORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "tidak ditemukan")
CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
DEVICE="${TORCH_DEVICE:-cpu}"

echo "  Target device  : ${DEVICE}"
echo "  Python         : ${PYTHON_VER}"
echo "  PyTorch        : ${PYTORCH_VER}"
echo "  CUDA tersedia  : ${CUDA_AVAIL}"
echo "  Working dir    : $(pwd)"
echo ""

# ── Validasi CUDA jika target adalah cuda ────────────────────────────────────
if [ "${DEVICE}" = "cuda" ] && [ "${CUDA_AVAIL}" != "True" ]; then
    echo "⚠️  WARNING: TORCH_DEVICE=cuda tapi CUDA tidak tersedia."
    echo "   Pastikan container dijalankan dengan flag --gpus all"
    echo "   Contoh: docker run --gpus all thesis-sst:cuda"
    echo ""
fi

# ── Validasi direktori data ───────────────────────────────────────────────────
if [ ! -d "/workspace/data" ]; then
    echo "⚠️  WARNING: /workspace/data tidak ditemukan."
    echo "   Pastikan volume di-mount: -v ./data:/workspace/data"
    echo ""
fi

# ── Validasi kredensial CMEMS (jika diperlukan) ───────────────────────────────
if [ -z "${COPERNICUSMARINE_USERNAME:-}" ] && [ -z "${COPERNICUSMARINE_SERVICE_USERNAME:-}" ]; then
    echo "ℹ️  INFO: COPERNICUSMARINE_USERNAME tidak di-set."
    echo "   Set di file .env jika ingin mengunduh data CMEMS."
    echo ""
fi

echo "────────────────────────────────────────────────────────────────"
echo ""

# ── Eksekusi perintah yang diberikan ─────────────────────────────────────────
exec "$@"
