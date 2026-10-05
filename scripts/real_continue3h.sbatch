#!/bin/bash
#SBATCH --job-name=real-continue3h
#SBATCH --output=logs/real_continue3h_%j.out
#SBATCH --error=logs/real_continue3h_%j.err
#SBATCH --partition=monakhova
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000&avx2
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=03:00:00

set -euo pipefail
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/convrml"
export CONVRML_DATA_PATH=/share/monakhova/shamus_data/parallel_100k
cd "$HOME/ConvRML"
python -u - <<'CHECK'
from omegaconf import OmegaConf
from dataset import get_loader
c=OmegaConf.load('configs/PLD_RML/config_real_continue3h.yaml')
train,val,test=get_loader(c)
y,x,n=train.dataset[0]
print('PREFLIGHT',n,tuple(y.shape),tuple(x.shape),y.min().item(),y.max().item(),x.min().item(),x.max().item(),flush=True)
assert tuple(y.shape)==tuple(x.shape)==(3,300,480)
CHECK
start_seconds=$SECONDS
python -u train_quantile.py --config configs/PLD_RML/config_real_continue3h.yaml

mkdir -p results/real_continue3h
printf "%s\n" "$((SECONDS-start_seconds))" > results/real_continue3h/training_seconds.txt
python -u scripts/evaluate_real_continue3h.py

EVAL_CHECKPOINT=checkpoints/real_rml_uniform_1024_gn/best_model.pth EVAL_OUTPUT=results/real_continue3h_baseline python -u scripts/evaluate_real_continue3h.py
