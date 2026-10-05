#!/bin/bash
#SBATCH --job-name=rml-four50
#SBATCH --output=logs/four50_%A_%a.out
#SBATCH --error=logs/four50_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --exclude=nikola-compute-15
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%6
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/four_models50.py "$@"
