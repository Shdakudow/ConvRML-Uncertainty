#!/bin/bash
#SBATCH --job-name=rml-residual-cdf
#SBATCH --output=logs/residual_cdf_%j.out
#SBATCH --error=logs/residual_cdf_%j.err
#SBATCH --partition=monakhova
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$HOME/.conda/envs/convrml/bin/python" -u scripts/residual_distribution_experiment.py
