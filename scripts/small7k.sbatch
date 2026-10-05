#!/bin/bash
#SBATCH --job-name=rml-5k-crps
#SBATCH --output=logs/small7k_%A_%a.out
#SBATCH --error=logs/small7k_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --exclude=nikola-compute-15
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [ "$1" = "infer" ]; then
    "$HOME/.conda/envs/convrml/bin/python" -u scripts/small7k_eval.py "$@"
else
    "$HOME/.conda/envs/convrml/bin/python" -u scripts/small7k_train.py "$@"
fi
