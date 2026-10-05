#!/bin/bash
#SBATCH --job-name=rml-quantile-report
#SBATCH --output=logs/rml_quantile_report_%j.out
#SBATCH --error=logs/rml_quantile_report_%j.err
#SBATCH --partition=monakhova
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/formal_quantile_evaluate.py report
