# Gaussian-only specialists

Train Im2im and QUTCC for 50 epochs using the same shared trainer and model
losses. Every measurement receives fresh Gaussian noise with SD 0.01 on the
[0,1] scale, followed by clipping. Targets remain unchanged.

For a controlled comparison, supply the generalized run's **untrained epoch-zero
weights** for the same method. The trainer checks the zero-step metadata and
strictly loads the tensors; it does not load trained generalized weights.
Recorded hashes in [protocol.json](protocol.json) identify the historical
comparison's initial states. New matched runs should use the same reference
file for each method in both studies.

```bash
python experiments/generalized_inputs/train.py \
  --protocol experiments/gaussian_specialist/protocol.json \
  --method qutcc --manifest /path/to/split.json \
  --initial-random-state /path/to/generalized/qutcc/initial_random.pt \
  --output runs/gaussian/qutcc --epochs 50 --device cuda
```

Repeat for Im2im using its own reference and output folder. Adam starts fresh.
The first 50 epochs' learning-rate schedule, batch size, seed and validation
panel match the generalized study. Resume complete epochs with `--resume`
and omit `--initial-random-state`. Calibration and test evaluation are separate.

A specialist receives roughly ten times as many Gaussian examples as the mixture
at the same epoch count; report that exposure difference when comparing scores.
Historical launchers and stronger deployment identity checks are preserved in
[the snapshot](../../archive/research_prototype_20261102/).
