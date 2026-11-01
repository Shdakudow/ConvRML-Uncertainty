# Gaussian-only specialists: 50 epochs from scratch

Train one QUTCC and one Im2im on all 17,999 original MIRFLICKR training pairs.
Every measurement receives independent additive Gaussian noise with **SD 0.01**
on the [0,1] intensity scale, followed by clipping. Targets stay unchanged. There
is no clean replay, no other modification, no PNG quantization, and no saved
modified dataset. Noise is generated anew during training.

Both models use the generalized comparison's exact **untrained epoch-zero
weights**, verified by method, seed, zero trained steps, and full weight hash.
These are random initial states, not trained generalized or clean checkpoints.
Adam starts fresh. The architectures, objectives, seed, batch order, effective
batch four, float32 precision, and 50-epoch learning-rate schedule match the
generalized models' first 50 epochs: nine-epoch warmup to 1e-4, then cosine to
1e-9. Model snapshots are saved every ten epochs and at epoch 50. Frequent full
recovery checkpoints retain Adam and all RNG states.

The shared training engine is
[../generalized_inputs/train.py](../generalized_inputs/train.py), configured by
[protocol.json](protocol.json). There is no copied training loop. Validation uses
the same fixed 100 scenes under all ten conditions, including the exact same
Gaussian noise draws as the generalized run. Gaussian validation SSIM/PSNR are
the primary comparison; clean and other conditions reveal specialization costs.
Calibration and test images are not loaded by this experiment.

The comparison matches **optimizer steps and training scenes** at epoch 50. A
specialist sees roughly ten times as many Gaussian examples as the generalized
mixture at the same epoch budget. A gain therefore does not by itself isolate
model-capacity interference from greater Gaussian exposure.

From a frozen repository snapshot containing `source_commit.txt`:

```bash
python experiments/gaussian_specialist/submit.py \
  --run-root /path/to/new/gaussian-specialists \
  --project /home/jy2228/ConvRML \
  --comparison-root /path/to/original/generalized-run
```

Each full GPU job depends on its own smoke test. Smoke updates are discarded;
full training starts again at epoch zero. Results stay in a new run directory;
the generalized runs and their checkpoints remain separate.
