#!/bin/bash
#SBATCH --job-name=convrml-baseline-report
#SBATCH --output=logs/convrml_baseline_report_%j.out
#SBATCH --error=logs/convrml_baseline_report_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=avx2&(gpu-mid|gpu-high)
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"
cd "$HOME/ConvRML"
python scripts/render_baseline_replication_pdf.py
