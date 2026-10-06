#!/bin/bash
#SBATCH --job-name=toy-uniform
#SBATCH --output=logs/toy_uniform_%j.out
#SBATCH --error=logs/toy_uniform_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=6000ada&avx2
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:40:00

set -euo pipefail
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"
export CONVRML_DATA_PATH=/share/monakhova/shamus_data/parallel_100k
cd "$HOME/ConvRML"
start_seconds=$SECONDS
python -u train_quantile.py --config configs/PLD_RML/config_quantile_toy_uniform.yaml

mkdir -p results/toy_uniform
printf "%s\n" "$((SECONDS-start_seconds))" > results/toy_uniform/training_seconds.txt
python -u scripts/diagnose_uniform.py
