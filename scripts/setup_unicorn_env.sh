#!/bin/bash
# Run inside a short Unicorn Slurm allocation, not on the login node.
set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
if [ ! -x "$HOME/.conda/envs/convrml/bin/python" ]; then
    conda create -y --prefix "$HOME/.conda/envs/convrml" python=3.11
fi
conda activate "$HOME/.conda/envs/convrml"

cd "$HOME/ConvRML"
# ConvRML was developed with the CUDA 11.8 build. Unicorn's CUDA 12.8 driver
# supports this older runtime and it remains compatible with older CPU nodes.
python -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-cache-dir -r requirements.txt
python -c 'import torch, dataset, train; print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")'
