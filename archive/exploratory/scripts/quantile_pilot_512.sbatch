#!/bin/bash
#SBATCH --job-name=convrml-q-pilot
#SBATCH --output=logs/convrml_q_pilot_%j.out
#SBATCH --error=logs/convrml_q_pilot_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=6000ada&avx2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"
export CONVRML_DATA_PATH=/share/monakhova/shamus_data/parallel_100k
cd "$HOME/ConvRML"
python train_quantile.py --config configs/PLD_RML/config_quantile_rml_pilot_512.yaml
