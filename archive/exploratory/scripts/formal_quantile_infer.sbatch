#!/bin/bash
#SBATCH --job-name=rml-quantile-infer
#SBATCH --output=logs/rml_quantile_infer_%j.out
#SBATCH --error=logs/rml_quantile_infer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/formal_quantile_evaluate.py infer --kind "$1"
