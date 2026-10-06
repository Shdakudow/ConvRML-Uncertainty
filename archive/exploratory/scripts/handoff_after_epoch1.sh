#!/bin/bash
# Save the completed first-epoch checkpoint and stop the training job.
source_job="$1"
project_dir="$HOME/ConvRML"
checkpoint="$project_dir/checkpoints/convnext_1.0_rml_mse_6_x4_downsize/latest_model.pth"
saved_checkpoint="$project_dir/checkpoints/convnext_1.0_rml_mse_6_x4_downsize/epoch_001_model.pth"
result_file="$project_dir/checkpoints/convnext_1.0_rml_mse_6_x4_downsize/FIRST_EPOCH_RESULT.txt"
training_log="$project_dir/logs/convrml_${source_job}.err"

# Unicorn's profile references optional terminal variables, so load it before
# enabling the strict unset-variable check.
source /etc/profile
set -euo pipefail
while [ ! -s "$checkpoint" ]; do
    if ! squeue -h -j "$source_job" | grep -q .; then
        echo "Source job $source_job ended before it wrote a checkpoint; stopping."
        exit 1
    fi
    sleep 60
done

cp -f "$checkpoint" "$saved_checkpoint"
{
    echo "First epoch checkpoint saved: $(date -Is)"
    echo "Source Slurm job: $source_job"
    echo "Checkpoint: $saved_checkpoint"
    shasum -a 256 "$saved_checkpoint"
    echo "Validation metrics:"
    grep "Val Loss:" "$training_log" | tail -1 || echo "Validation metrics were not found in $training_log"
} > "$result_file"
echo "Saved: $saved_checkpoint"
echo "Record: $result_file"
scancel "$source_job"
