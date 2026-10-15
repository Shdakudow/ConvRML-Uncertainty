#!/bin/bash
#SBATCH --job-name=adaptive-q50
#SBATCH --output=logs/adaptive_q50_%j.out
#SBATCH --error=logs/adaptive_q50_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=(h100|h200|6000ada|a6000)&avx2
#SBATCH --prefer=h100|h200|6000ada
#SBATCH --exclude=monakhova-compute-01,nikola-compute-15,bhattacharjee-compute-02
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the experiment folder}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# Forward advance notice to Python so it checkpoints at a batch boundary.
child=''
trap 'if [ -n "$child" ]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1
"$HOME/.conda/envs/convrml/bin/python" -u scripts/train_constrained_adaptive50.py "$@" &
child=$!
set +e
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; status=$?; fi
if [ "$status" -eq 75 ]; then
    /usr/local/slurm/current/bin/scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
