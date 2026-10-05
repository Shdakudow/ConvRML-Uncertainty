#!/bin/bash
#SBATCH --job-name=convrml
#SBATCH --output=logs/convrml_%j.out
#SBATCH --error=logs/convrml_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=avx2&(gpu-mid|gpu-high)
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --requeue

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"

export CONVRML_DATA_PATH=/share/monakhova/shamus_data/parallel_100k
cd "$HOME/ConvRML"
mkdir -p logs
python train.py --config configs/PLD_RML/config_local_tiny_tuned.yaml
