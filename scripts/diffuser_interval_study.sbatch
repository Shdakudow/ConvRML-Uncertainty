#!/bin/bash
#SBATCH --job-name=diffuser60k
#SBATCH --output=logs/diffuser60k_%A_%a.out
#SBATCH --error=logs/diffuser60k_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --exclude=nikola-compute-15,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint=(h100|h200|6000ada)&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/diffuser_interval_study.py "$@" --worker "$SLURM_ARRAY_TASK_ID"
