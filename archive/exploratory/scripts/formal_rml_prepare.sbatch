#!/bin/bash
#SBATCH --job-name=rml25k-prepare
#SBATCH --output=logs/rml25k_prepare_%j.out
#SBATCH --error=logs/rml25k_prepare_%j.err
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/prepare_formal_rml.py
