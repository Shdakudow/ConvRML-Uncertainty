#!/bin/bash
#SBATCH --job-name=rml25k-report20
#SBATCH --output=logs/rml25k_report20_%j.out
#SBATCH --error=logs/rml25k_report20_%j.err
#SBATCH --partition=default_partition
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/aggregate_formal_rml.py --root results/formal_rml_25k_epoch20
