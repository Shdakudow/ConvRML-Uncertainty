#!/bin/bash
#SBATCH --job-name=rml-q-followup
#SBATCH --output=logs/q_followup_%j.out
#SBATCH --error=logs/q_followup_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/gradient_conflict_experiment.py "$@"
