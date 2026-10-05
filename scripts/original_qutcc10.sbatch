#!/bin/bash
#SBATCH --job-name=original-qutcc10
#SBATCH --output=logs/original_qutcc10_%A_%a.out
#SBATCH --error=logs/original_qutcc10_%A_%a.err
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
smoke|train) "$PY" -u scripts/original_qutcc10_study.py "$stage" --task "$SLURM_ARRAY_TASK_ID" "$@" ;;
tune|evaluate) "$PY" -u scripts/original_qutcc10_evaluate.py "$stage" --task "$SLURM_ARRAY_TASK_ID" ;;
select)
 camera=rml; if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then camera=diffuser; fi
 "$PY" -u scripts/original_qutcc10_evaluate.py select --camera "$camera" ;;
esac
