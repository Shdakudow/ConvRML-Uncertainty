#!/bin/bash
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/snapshot_%j.out
#SBATCH --error=logs/snapshot_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/jy2228/.conda/envs/convrml/bin/python -u export_raw_demo.py --snapshot
