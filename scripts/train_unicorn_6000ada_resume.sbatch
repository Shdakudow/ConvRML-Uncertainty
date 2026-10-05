#!/bin/bash
#SBATCH --job-name=convrml-ada
#SBATCH --output=logs/convrml_ada_%j.out
#SBATCH --error=logs/convrml_ada_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=6000ada&avx2
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4-12:00:00
#SBATCH --requeue

set -euo pipefail
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"
export CONVRML_DATA_PATH=/share/monakhova/shamus_data/parallel_100k
cd "$HOME/ConvRML"
python train.py --config configs/PLD_RML/config_unicorn_6000ada_resume_17.yaml
