#!/bin/bash
#SBATCH --job-name=rml-momentum-cdf
#SBATCH --output=logs/momentum_cdf_%j.out
#SBATCH --error=logs/momentum_cdf_%j.err
#SBATCH --partition=monakhova
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$HOME/.conda/envs/convrml/bin/python" -u scripts/momentum_residual_experiment.py
