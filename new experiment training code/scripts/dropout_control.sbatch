#!/bin/bash
#SBATCH --job-name=dropout-zero-control
#SBATCH --output=logs/dropout_zero_%j.out
#SBATCH --error=logs/dropout_zero_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=(h100|h200|6000ada|a6000)&avx2
#SBATCH --prefer=h100|h200|6000ada
#SBATCH --exclude=monakhova-compute-01,nikola-compute-15,bhattacharjee-compute-02
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/train_dropout_control.py "$@"
