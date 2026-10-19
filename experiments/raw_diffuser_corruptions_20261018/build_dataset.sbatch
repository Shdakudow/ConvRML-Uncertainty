#!/bin/bash
#SBATCH --partition=default_partition
#SBATCH --job-name=raw-corruption-data-only
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dataset_%j.out
#SBATCH --error=logs/dataset_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/jy2228/.conda/envs/convrml/bin/python -u build_dataset.py --workers 4
