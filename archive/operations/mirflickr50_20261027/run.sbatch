#!/bin/bash
# Resource requests are supplied by submit.py, separately for audit and GPU jobs.
set -euo pipefail
source_dir=$1
study=$2
manifest=$3
action=$4
method=${5:-}
target_epoch=${6:-}
python=/home/jy2228/.conda/envs/convrml/bin/python
operations="$source_dir/archive/operations/mirflickr50_20261027"
cd "$source_dir"
export PYTHONPATH="$source_dir"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

case "$action" in
  audit)
    exec "$python" "$operations/preflight.py" data --manifest "$manifest" \
      --output "$study/preflight/data.json"
    ;;
  model)
    exec "$python" "$operations/preflight.py" "$method" --manifest "$manifest" \
      --output "$study/preflight/$method.json"
    ;;
  train)
    "$python" -c 'import json,sys; assert all(json.load(open(path))["passed"] for path in sys.argv[1:])' \
      "$study/preflight/data.json" "$study/preflight/$method.json"
    args=(--dataset raw_clean --method "$method" --manifest "$manifest"
          --output "$study/results/$method" --epochs "$target_epoch"
          --device cuda --workers 2)
    if test -f "$study/results/$method/latest.pt"; then args+=(--resume); fi
    exec "$python" -u train.py "${args[@]}"
    ;;
  *)
    echo "Unknown action: $action" >&2
    exit 2
    ;;
esac
