#!/bin/bash
#SBATCH --account=monakhova
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=(6000ada|a6000)&avx2
#SBATCH --prefer=6000ada
#SBATCH --exclude=monakhova-compute-01,nikola-compute-15,bhattacharjee-compute-02
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
child=''
trap 'if [ -n "$child" ]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1
# Cancellation stops this run; only the checkpoint signal requests continuation.
trap 'if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi' TERM
"/home/jy2228/.conda/envs/convrml/bin/python" -u "$@" &
child=$!
set +e
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then
    wait "$child"
    status=$?
fi
if [ "$status" -eq 75 ]; then
    /usr/local/slurm/current/bin/scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
