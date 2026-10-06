#!/bin/bash
#SBATCH --job-name=pair-audit
#SBATCH --output=logs/pair_audit_%j.out
#SBATCH --error=logs/pair_audit_%j.err
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
set -eo pipefail
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"
export CONVRML_DATA_PATH=/share/monakhova/shamus_data/parallel_100k
cd "$HOME/ConvRML"
python -u scripts/audit_real_pairs.py
