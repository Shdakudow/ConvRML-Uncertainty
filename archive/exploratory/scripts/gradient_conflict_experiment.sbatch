#!/bin/bash
#SBATCH --job-name=rml-gradient-test
#SBATCH --output=logs/gradient_test_%A_%a.out
#SBATCH --error=logs/gradient_test_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/gradient_conflict_experiment.py "$@"
