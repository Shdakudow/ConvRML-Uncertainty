#!/bin/bash
#SBATCH --job-name=rml-5k-cal95
#SBATCH --output=logs/small7k_cal95_%A_%a.out
#SBATCH --error=logs/small7k_cal95_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --exclude=nikola-compute-15
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-5,7-14%2
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/small7k_calibrate95.py infer
