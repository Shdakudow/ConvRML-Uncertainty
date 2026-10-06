#!/bin/bash
#SBATCH --job-name=rcps-comparison
#SBATCH --output=logs/rcps_comparison_%A_%a.out
#SBATCH --error=logs/rcps_comparison_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --exclude=nikola-compute-15,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint=(h100|h200|6000ada)&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this experiment directory}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/compare_rcps.py --worker "$SLURM_ARRAY_TASK_ID" "$@"
