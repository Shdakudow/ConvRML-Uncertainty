#!/bin/bash
#SBATCH --job-name=real-small
#SBATCH --output=logs/real_small_%j.out
#SBATCH --error=logs/real_small_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=6000ada&avx2
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:00:00

set -euo pipefail
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"
export CONVRML_DATA_PATH=/share/monakhova/shamus_data/parallel_100k
cd "$HOME/ConvRML"
python -u - <<'CHECK'
from omegaconf import OmegaConf
from dataset import get_loader
c=OmegaConf.load('configs/PLD_RML/config_real_small.yaml')
train,val,test=get_loader(c)
y,x,n=train.dataset[0]
print('PREFLIGHT',n,tuple(y.shape),tuple(x.shape),y.min().item(),y.max().item(),x.min().item(),x.max().item(),flush=True)
assert tuple(y.shape)==tuple(x.shape)==(3,300,480)
CHECK
start_seconds=$SECONDS
python -u train_quantile.py --config configs/PLD_RML/config_real_small.yaml

mkdir -p results/real_small
printf "%s\n" "$((SECONDS-start_seconds))" > results/real_small/training_seconds.txt
python -u scripts/evaluate_real_small.py
