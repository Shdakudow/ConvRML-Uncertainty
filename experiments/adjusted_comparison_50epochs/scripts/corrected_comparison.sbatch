#!/bin/bash
#SBATCH --job-name=corrected-convrml
#SBATCH --output=logs/corrected_%j.out
#SBATCH --error=logs/corrected_%j.err
#SBATCH --partition=gpu
#SBATCH --exclude=nikola-compute-15,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint=(h100|h200|6000ada)&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from experiment root}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="$HOME/.conda/envs/convrml/bin/python"
stage="$1";shift
case "$stage" in
smoke|train) "$PY" -u scripts/corrected_comparison_study.py "$stage" "$@" ;;
tune|select|evaluate) "$PY" -u scripts/corrected_comparison_evaluate.py "$stage" "$@" ;;
esac
