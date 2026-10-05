#!/bin/bash
#SBATCH --job-name=convnext_normfilm10-study
#SBATCH --output=logs/convnext_normfilm10_%A_%a.out
#SBATCH --error=logs/convnext_normfilm10_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --exclude=nikola-compute-15,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint=(h100|h200|6000ada)&avx2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
set -euo pipefail
cd "$HOME/ConvRML"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="$HOME/.conda/envs/convrml/bin/python"
stage="$1";shift
case "$stage" in
smoke|train) "$PY" -u scripts/convnext_normfilm10_study.py "$stage" --task "$SLURM_ARRAY_TASK_ID" "$@" ;;
head-smoke|head-train)
 camera=rml; if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then camera=diffuser; fi
 "$PY" -u scripts/convnext_normfilm10_study.py "${stage#head-}" --camera "$camera" --arm fixed "$@" ;;
tune|evaluate) "$PY" -u scripts/convnext_normfilm10_evaluate.py "$stage" --task "$SLURM_ARRAY_TASK_ID" ;;
select)
 camera=rml; if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then camera=diffuser; fi
 "$PY" -u scripts/convnext_normfilm10_evaluate.py select --camera "$camera" ;;
esac
