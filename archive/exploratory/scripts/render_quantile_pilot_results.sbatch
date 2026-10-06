#!/bin/bash
#SBATCH --job-name=convrml-q-render
#SBATCH --output=logs/convrml_q_render_%j.out
#SBATCH --error=logs/convrml_q_render_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=6000ada&avx2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"
export CONVRML_DATA_PATH=/share/monakhova/shamus_data/parallel_100k
cd "$HOME/ConvRML"
python scripts/render_quantile_pilot_results.py \
  --config configs/PLD_RML/config_quantile_rml_pilot_4096.yaml \
  --checkpoint checkpoints/quantile_convrml_tiny_pilot_4096/best_model.pth \
  --output-dir results/quantile_convrml_tiny_pilot_4096/evaluation_10 \
  --num-samples 10
