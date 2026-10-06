#!/bin/bash
#SBATCH --job-name=rml25k-baselines
#SBATCH --output=logs/rml25k_train_%A_%a.out
#SBATCH --error=logs/rml25k_train_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/formal_rml_train.py
