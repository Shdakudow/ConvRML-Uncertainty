#!/bin/bash
#SBATCH --job-name=grid-continue
#SBATCH --output=logs/toy_grid_continue_%j.out
#SBATCH --error=logs/toy_grid_continue_%j.err
#SBATCH --partition=monakhova
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/toy_grid_continue.py
