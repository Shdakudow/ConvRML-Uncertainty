#!/bin/bash
#SBATCH --job-name=motion-pair
#SBATCH --output=logs/motion_pair_%j.out
#SBATCH --error=logs/motion_pair_%j.err
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
for method in endpoint uniform; do
    start_seconds=$SECONDS
    python -u train_quantile.py --config "configs/PLD_RML/config_motion_${method}.yaml"
    mkdir -p "results/motion_${method}"
    printf "%s\n" "$((SECONDS-start_seconds))" > "results/motion_${method}/training_seconds.txt"
    python -u scripts/evaluate_motion.py --method "$method"
done
