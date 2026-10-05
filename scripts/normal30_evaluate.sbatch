#!/bin/bash
#SBATCH --job-name=rml-normal30-eval
#SBATCH --output=logs/normal30_eval_%j.out
#SBATCH --error=logs/normal30_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/normal30_experiment.py evaluate "$@"
