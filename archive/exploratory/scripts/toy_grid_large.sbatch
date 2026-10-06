#!/bin/bash
#SBATCH --job-name=toy-grid-large
#SBATCH --output=logs/toy_grid_large_%j.out
#SBATCH --error=logs/toy_grid_large_%j.err
#SBATCH --partition=monakhova
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/toy_grid_compare.py --train-count 8192 --val-count 1024 --test-count 1024 --epochs 8 --offset 32000 --out results/toy_grid_large
