#!/bin/bash
# =============================================================================
# docker/slurm_job.sh — Slurm job script untuk HPC BMKG
# Jalankan: sbatch docker/slurm_job.sh
#
# Spesifikasi HPC BMKG yang ditarget:
#   16 vCPU, 32 GB RAM, 1 GPU (16 GB VRAM), Ubuntu 22.04, CUDA 12.1
# =============================================================================

#SBATCH --job-name=thesis-sst-training
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # Sesuaikan dengan nama partisi GPU di HPC BMKG

# ── Setup ─────────────────────────────────────────────────────────────────────
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo ""

mkdir -p logs models/checkpoints reports

SIF_PATH="${HOME}/thesis-sst/thesis-sst.sif"

if [ ! -f "${SIF_PATH}" ]; then
    echo "ERROR: ${SIF_PATH} tidak ditemukan."
    echo "Jalankan build_for_hpc.sh dan upload .sif terlebih dahulu."
    exit 1
fi

# ── Full hyperparameter search: 4 model × 4 lookback × 4 horizon ─────────────
echo "Memulai hyperparameter search (64 run)..."
echo ""

for LOOKBACK in 7 14 21 30; do
    for MODEL in lstm transformer hybrid; do
        for HORIZON in 1 3 7 14; do
            echo "──────────────────────────────────────────────"
            echo "Training: model=${MODEL} | lookback=${LOOKBACK} | h=${HORIZON}"
            echo "──────────────────────────────────────────────"

            singularity exec \
                --nv \
                --bind "${PWD}/data:/workspace/data" \
                --bind "${PWD}/models:/workspace/models" \
                --bind "${PWD}/configs:/workspace/configs" \
                --bind "${PWD}/reports:/workspace/reports" \
                --bind "${PWD}/logs:/workspace/logs" \
                "${SIF_PATH}" \
                python src/models/train.py \
                    --model "${MODEL}" \
                    --lookback "${LOOKBACK}" \
                    --horizon "${HORIZON}" \
                    --device cuda \
                    --config configs/model_config.yaml \
                    --checkpoint-dir models/checkpoints

            echo "✓ Selesai: ${MODEL}_lb${LOOKBACK}_h${HORIZON}"
            echo ""
        done
    done
done

# ── Evaluasi akhir ────────────────────────────────────────────────────────────
echo "Menjalankan evaluasi komparatif..."

singularity exec \
    --nv \
    --bind "${PWD}/data:/workspace/data" \
    --bind "${PWD}/models:/workspace/models" \
    --bind "${PWD}/reports:/workspace/reports" \
    "${SIF_PATH}" \
    python src/evaluation/evaluate.py \
        --config configs/model_config.yaml \
        --checkpoint-dir models/checkpoints \
        --output-dir reports

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✓ Semua training selesai."
echo "  End time: $(date)"
echo "  Hasil tersimpan di: reports/"
