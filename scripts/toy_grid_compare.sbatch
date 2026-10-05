#!/bin/bash
#SBATCH --job-name=toy-grid-compare
#SBATCH --output=logs/toy_grid_compare_%j.out
#SBATCH --error=logs/toy_grid_compare_%j.err
#SBATCH --partition=monakhova
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/toy_grid_compare.py
