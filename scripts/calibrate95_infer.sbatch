#!/bin/bash
#SBATCH --job-name=rml-cal95-infer
#SBATCH --output=logs/cal95_infer_%A_%a.out
#SBATCH --error=logs/cal95_infer_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=a6000&avx2
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/calibrate_formal95.py infer
