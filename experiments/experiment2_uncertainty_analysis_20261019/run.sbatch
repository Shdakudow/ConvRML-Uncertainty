#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=(h100|h200|6000ada|a6000|a40|a100|l40s|3090)&avx2
#SBATCH --exclude=monakhova-compute-01,nikola-compute-15,bhattacharjee-compute-02
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$HOME/.conda/envs/convrml/bin/python" -u experiments/experiment2_uncertainty_analysis_20261019/evaluate_full_test.py --camera "$1" --method "$2" --epoch "${3:-50}" --smoke
"$HOME/.conda/envs/convrml/bin/python" -u experiments/experiment2_uncertainty_analysis_20261019/evaluate_full_test.py --camera "$1" --method "$2" --epoch "${3:-50}"
