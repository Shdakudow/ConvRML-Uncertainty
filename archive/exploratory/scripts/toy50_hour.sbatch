#!/bin/bash
#SBATCH --job-name=rml-toy50
#SBATCH --output=logs/toy50_%j.out
#SBATCH --error=logs/toy50_%j.err
#SBATCH --partition=gpu
#SBATCH --exclude=nikola-compute-15
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/toy50_hour.py "$@"
