#!/bin/bash
#SBATCH --job-name=thesis-sst-download
#SBATCH --output=logs/download_%j.log
#SBATCH --error=logs/download_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --partition=cpu          # download tidak butuh GPU

# ===========================================================================
# Job Script khusus download data CMEMS
# Pastikan node punya akses internet (tanyakan admin HPC BMKG)
# ===========================================================================

WORKDIR=$HOME/thesis-sst
CONTAINER=$HOME/thesis-sst.sif

cd $WORKDIR
mkdir -p logs data/raw

echo "Download CMEMS dimulai: $(date)"

singularity exec \
    --bind $WORKDIR:/workspace \
    --bind $HOME/.copernicusmarine:/root/.copernicusmarine \
    $CONTAINER \
    python -m src.data.download \
    --config /workspace/configs/config.yaml

echo "Download selesai: $(date)"
