#!/bin/bash
#SBATCH --job-name=rml-four-1000-a10
#SBATCH --output=logs/rml_four_1000_%j.out
#SBATCH --error=logs/rml_four_1000_%j.err
#SBATCH --partition=monakhova
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$HOME/.conda/envs/convrml/bin/python" -u scripts/evaluate_four_1000_alpha10.py
