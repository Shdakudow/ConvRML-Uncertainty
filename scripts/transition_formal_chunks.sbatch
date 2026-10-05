#!/bin/bash
#SBATCH --job-name=rml25k-chunk-switch
#SBATCH --output=logs/rml25k_chunk_switch_%j.out
#SBATCH --error=logs/rml25k_chunk_switch_%j.err
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=03:10:00
set -euo pipefail
cd "$HOME/ConvRML"
"$HOME/.conda/envs/convrml/bin/python" -u scripts/transition_formal_chunks.py
