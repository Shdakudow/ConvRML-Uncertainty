#!/bin/bash
#SBATCH --job-name=rml-cal95-report
#SBATCH --output=logs/cal95_report_%j.out
#SBATCH --error=logs/cal95_report_%j.err
#SBATCH --partition=monakhova
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/calibrate_formal95.py report
