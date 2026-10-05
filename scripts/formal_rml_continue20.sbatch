#!/bin/bash
#SBATCH --job-name=rml25k-to20
#SBATCH --output=logs/rml25k_to20_%A_%a.out
#SBATCH --error=logs/rml25k_to20_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/.conda/envs/convrml/bin/python" -u scripts/formal_rml_train.py --epochs 20 --patience 0 --resume-source results/formal_rml_25k --out results/formal_rml_25k_epoch20
