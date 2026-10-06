#!/bin/bash
#SBATCH --job-name=rml25k-smoke
#SBATCH --output=logs/rml25k_smoke_%A_%a.out
#SBATCH --error=logs/rml25k_smoke_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0,5%2
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/formal_rml_train.py --smoke
