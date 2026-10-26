# Train and evaluate

Commands run from a clone after `python -m pip install -e . -c reproducibility/constraints-tested.txt`. They do not submit
cluster jobs. Full training requires suitable compute; CPU unit tests use small
synthetic paired fixtures. The portable CLI supports Linux/macOS.

## Prepare data paths

Set a path to your prepared data:

```bash
export CONVRML_DATA=/path/to/prepared-data
```

[Data availability and layout](../../../reproducibility/DATA.md) defines the two loaders.
Use the saved split IDs/order; training never uses calibration or test targets.
Formal runs validate the recorded counts and ordered-ID hashes. `--smoke` permits
explicitly labelled small fixtures and is not a full-dataset experiment.

## Download saved weights

```bash
python -m convrml.download --dataset rml --method qutcc --epoch 50 --output weights/rml/qutcc
python -m convrml.download --dataset rml --method im2im --epoch 50 --output weights/rml/im2im
python -m convrml.download --dataset rml --method mc_dropout --epoch 50 --output weights/rml/mc
python -m convrml.download --dataset rml --method ensemble --epoch 50 --members 5 --output weights/rml/ensemble
python -m convrml.download --dataset raw_clean --method qutcc --epoch 50 --output weights/raw/qutcc
```

`--list` shows names/hashes without downloading. Ensemble downloads default to
members 0–4; `--members 10` selects 0–9 and `--member N` selects one. Every file
is checked against the [index](../../../reproducibility/checkpoints.json). Corrupted raw
uses adaptation epochs 5–30; clean models use epochs 5–50. A private repository
requires an authenticated GitHub CLI with access, or manually supplied indexed files.

## Evaluate RML / normalized DiffuserCam

```bash
python infer.py --dataset rml --method qutcc \
  --checkpoints weights/rml/qutcc/exp2__rml__qutcc__epoch_050.pt \
  --manifest "$CONVRML_DATA/rml/manifest.json" --pairs "$CONVRML_DATA/rml/pairs.npy" \
  --calibration-manifest "$CONVRML_DATA/rml/calibration/manifest.json" \
  --calibration-pairs "$CONVRML_DATA/rml/calibration/pairs.npy" \
  --output outputs/paper/rml/qutcc/epoch_050 --examples 10 --device cuda
```

Use the same arguments and dataset IDs with the other methods, changing the
checkpoint(s) and output folder. For ensemble, pass members 0–4 or 0–9 **in order**
to `--checkpoints`; mixed epochs or repeated initialization/data seeds are rejected.
MC dropout uses p=0.05 with 30 stochastic passes. Its default inference seed
matches the captured predictor's initialization seed; `--seed` can explicitly
select another draw sequence. Seed, batch shape and hardware affect stochastic results. `diffuser` selects the normalized
parallel-camera data, distinct from raw DLMD.

## Evaluate raw DiffuserCam

```bash
python infer.py --dataset raw_clean --method qutcc \
  --checkpoints weights/raw/qutcc/exp3__raw_clean__qutcc__epoch_050.pt \
  --manifest "$CONVRML_DATA/raw_clean/split.json" \
  --calibration-manifest "$CONVRML_DATA/raw_clean/split.json" \
  --output outputs/paper/raw_clean/qutcc/epoch_050 --examples 10 --device cuda
```

Raw manifests contain separate roles in one file; the calibration/test role IDs
are disjoint. No `--pairs` argument is used. Replace method/checkpoint for raw
Im2im. `raw_corrupted` selects its own fixed-corruption manifest and checkpoint.
Models pad 270×480 internally to 288×480 and crop every output back.

## Train and resume

```bash
python train.py --dataset rml --method qutcc \
  --manifest "$CONVRML_DATA/rml/manifest.json" --pairs "$CONVRML_DATA/rml/pairs.npy" \
  --output runs/rml/qutcc --epochs 5 --device cuda
python train.py --dataset rml --method qutcc \
  --manifest "$CONVRML_DATA/rml/manifest.json" --pairs "$CONVRML_DATA/rml/pairs.npy" \
  --output runs/rml/qutcc --epochs 10 --resume --device cuda
```

`--epochs` is the absolute target. The learning-rate horizon remains the recorded
50 epochs, so chunking does not restart warmup/cosine decay. Checkpoints are saved
every five epochs, with full optimizer/RNG state in `latest.pt` after each complete
epoch. Resume validates source, software, data identity, device and worker count.
A preempted partial epoch restarts from the last completed epoch. The exact
historical mid-epoch resume engines remain archived.

Im2im/MC use their own recorded profiles. Ensemble training requires `--member N`
and a distinct output directory per member; each member sees the entire training
split with independent initialization and shuffle streams. Do not train concurrent
writers into the same output directory.

For raw clean training, use `--dataset raw_clean`, the image-file
manifest and no `--pairs`. Corruption adaptation uses `--dataset raw_corrupted`
and `--initialize` with that method's clean epoch-30 checkpoint. It starts fresh
Adam moments at LR 1e-5, then retains them through adaptation epoch 30. Existing
historical full-state checkpoints are not silently treated as portable resumes.

## Outputs

Evaluation writes `calibration.json` before opening test pixels, then `report.json`,
`per_image.csv`, and optional `examples.npz` with measurement, target, prediction,
lower and upper arrays; saved example predictions/bounds are clipped to [0,1]. Existing results are immutable: use a new output directory
for another checkpoint/protocol. Per-image metrics and means include LPIPS-AlexNet
(clipped images), Pearson/Spearman width/error correlations, SSIM, PSNR, MAE,
coverage, width and crossings. Undefined correlations and intentionally omitted
LPIPS are null, with valid-image counts. `--no-lpips` is available for tiny/offline
fixtures; formal evaluation defaults to pretrained LPIPS. Cache its weights before
running on an offline worker. Infeasible quantiles produce `failure.json`, not
zero-valued metrics. Alpha=0.10 targets 90% coverage; actual test coverage is measured.

Plot complete milestones with the optional plotting dependencies:

```bash
python -m pip install -e '.[plot]'
python archive/publication_support/tools/plot_learning_curves.py --series QUTCC=outputs/paper/rml/qutcc \
  --metric-space clipped --output figures/rml
```

This exports SSIM/PSNR/width/coverage in `learning_curves.png`,
LPIPS/Pearson/Spearman/MAE in `quality_curves.png` when available, and all values
in `learning_curves.csv`. It reads completed five-epoch reports; missing or
undefined values are gaps. Existing reports require evaluation to populate new
metrics; the plotter does not fabricate them or rerun inference.

[Settings and seeds](../../../configs/experiment_settings.json) ·
[Metrics](METHODS_AND_METRICS.md) · [Calibration](COVERAGE90.md)
