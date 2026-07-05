#!/bin/bash
#SBATCH --job-name=thesis-sst-hybrid
#SBATCH --output=logs/experiment_%j.log
#SBATCH --error=logs/experiment_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu          # GANTI sesuai nama partisi GPU di HPC BMKG

# ===========================================================================
# Job Script SLURM — thesis-sst
# Menjalankan 48 eksperimen (3 model x 4 lookback x 4 horizon)
# menggunakan container Singularity
# ===========================================================================

set -e    # hentikan jika ada error

# Direktori proyek dan container
WORKDIR=$HOME/thesis-sst
CONTAINER=$HOME/thesis-sst.sif

echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start      : $(date)"
echo "Workdir    : $WORKDIR"
echo "Container  : $CONTAINER"
echo "============================================"

cd $WORKDIR
mkdir -p logs results/figures data/raw data/processed

# -----------------------------------------------------------------------
# 1. Verifikasi GPU dalam container
# -----------------------------------------------------------------------
echo "[1/4] Verifikasi GPU..."
singularity exec --nv $CONTAINER python -c "
import torch
print('PyTorch :', torch.__version__)
print('CUDA    :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU     :', torch.cuda.get_device_name(0))
"

# -----------------------------------------------------------------------
# 2. Smoke test (validasi pipeline)
# -----------------------------------------------------------------------
echo "[2/4] Smoke test..."
singularity exec --nv \
    --bind $WORKDIR:/workspace \
    $CONTAINER \
    python /workspace/smoke_test.py

# -----------------------------------------------------------------------
# 3. Pra-pemrosesan data (jika belum ada)
# -----------------------------------------------------------------------
if [ ! -f "$WORKDIR/data/processed/sst_series.csv" ]; then
    echo "[3/4] Pra-pemrosesan data..."
    singularity exec --nv \
        --bind $WORKDIR:/workspace \
        $CONTAINER \
        python -m src.data.preprocess \
        --config /workspace/configs/config.yaml
else
    echo "[3/4] Data sudah terproses, skip pra-pemrosesan."
fi

# -----------------------------------------------------------------------
# 4. Jalankan 48 eksperimen
# -----------------------------------------------------------------------
echo "[4/4] Menjalankan 48 eksperimen..."
singularity exec --nv \
    --bind $WORKDIR:/workspace \
    $CONTAINER \
    python -m src.training.hyperparameter_search \
    --config /workspace/configs/config.yaml \
    --data /workspace/data/processed/sst_series.csv

# -----------------------------------------------------------------------
# 5. Analisis hasil
# -----------------------------------------------------------------------
echo "[5/5] Analisis dan visualisasi hasil..."
singularity exec --nv \
    --bind $WORKDIR:/workspace \
    $CONTAINER \
    python -m src.evaluation.visualize \
    --config /workspace/configs/config.yaml \
    --results /workspace/results/results_all.csv

echo "============================================"
echo "Selesai : $(date)"
echo "Hasil tersimpan di: $WORKDIR/results/"
echo "============================================"
