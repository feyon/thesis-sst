#!/bin/bash
#SBATCH --job-name=ocean-dl
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu_riset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

echo "================================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "================================================"

cd ~/thesis-sst
mkdir -p logs models/checkpoints reports data/processed

SIF_PATH="${HOME}/thesis-sst/thesis-sst.sif"

if [ ! -f "${SIF_PATH}" ]; then
    echo "ERROR: ${SIF_PATH} tidak ditemukan."
    exit 1
fi

# Cek GPU di compute node
echo "GPU info:"
apptainer exec --nv "${SIF_PATH}" python -c \
    "import torch; print('CUDA:', torch.cuda.is_available()); \
     print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo ""
echo "Memulai hyperparameter search (48 run)..."
echo ""

for LOOKBACK in 7 14 21 30; do
    for MODEL in lstm transformer hybrid; do
        for HORIZON in 1 3 7 14; do
            echo "Training: model=${MODEL} | lookback=${LOOKBACK} | h=${HORIZON}"
            apptainer exec --nv \
                --bind "${HOME}/thesis-sst/data:/workspace/data" \
                --bind "${HOME}/thesis-sst/models:/workspace/models" \
                --bind "${HOME}/thesis-sst/configs:/workspace/configs" \
                --bind "${HOME}/thesis-sst/reports:/workspace/reports" \
                --bind "${HOME}/thesis-sst/logs:/workspace/logs" \
                --bind "${HOME}/thesis-sst/src:/workspace/src" \
                "${SIF_PATH}" \
                python -m src.models.train \
                    --model "${MODEL}" \
                    --lookback "${LOOKBACK}" \
                    --horizon "${HORIZON}" \
                    --device cuda \
                    --config configs/config.yaml \
                    --checkpoint-dir models/checkpoints
            echo "✓ Selesai: ${MODEL}_lb${LOOKBACK}_h${HORIZON}"
        done
    done
done

echo ""
echo "================================================"
echo "Semua training selesai: $(date)"
echo "================================================"
