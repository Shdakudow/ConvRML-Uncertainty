#!/bin/bash
#SBATCH --job-name=film-ablation
#SBATCH --output=logs/film_ablation_%A_%a.out
#SBATCH --error=logs/film_ablation_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --exclude=nikola-compute-15,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint=(h100|h200|6000ada)&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
case "$SLURM_ARRAY_TASK_ID" in
  0) variant=control ;;
  1) variant=everywhere ;;
  *) exit 2 ;;
esac
"$HOME/.conda/envs/convrml/bin/python" -u scripts/film_finetune.py "$@" --variant "$variant"
