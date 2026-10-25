# Recorded experiment engines

This folder preserves the original Experiment 2/3 implementations and shared
source in their previous relative layout. The publication-facing models are in
[Experiment 2](../../experiments/experiment2/models/) and
[Experiment 3](../../experiments/experiment3/models/).

Use these engines for exact historical training/calibration/resume protocols;
[the recorded guide](docs/RUNNING_EXPERIMENTS.md) lists settings and external data
paths. Run commands relative to this folder when following its original paths.
The `tools` link points to the repository's maintained evaluation utilities.
Do not substitute a publication wrapper into a full-state historical resume:
its source-hash guard intentionally identifies the captured training source.

The relative model/shared-source links and source files are preserved. Moving
them in this repository does not alter cluster jobs, datasets or checkpoints.
