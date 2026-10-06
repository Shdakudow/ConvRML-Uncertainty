#!/bin/bash
#SBATCH --job-name=toy-report
#SBATCH --output=logs/toy_report_%j.out
#SBATCH --error=logs/toy_report_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=6000ada&avx2
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:10:00
set -eo pipefail
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"
cd "$HOME/ConvRML"
python -u scripts/diagnose_toy.py
