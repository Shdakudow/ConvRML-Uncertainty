#!/bin/bash
#SBATCH --job-name=convrml-toy-8k
#SBATCH --output=logs/convrml_toy_8k_%j.out
#SBATCH --error=logs/convrml_toy_8k_%j.err
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
python -u train_quantile.py --config configs/PLD_RML/config_quantile_toy_fourier.yaml
