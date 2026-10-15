# New experiment: image-adaptive QUTCC, without RCPS

This folder contains the current QUTCC training implementation for RML and DiffuserCam. It is separate from the older experiments; those files are unchanged. The snapshot was taken on October 15, 2026.

## What the model learns

The model predicts **one lower quantile and one upper quantile for each input image**. This pair is shared across pixels, but the resulting lower-bound and upper-bound images vary at every pixel. There is no per-pixel quantile selector in this experiment.

- **Reconstruction backbone:** Shamus ConvNeXtV2-Base with an FPN decoder, initialized randomly.
- **Quantile conditioning:** 48 FiLM sites, after normalization in 36 encoder residual blocks and 12 decoder convolution stages. A scalar q is embedded into 128 sinusoidal features and a 512-dimensional MLP embedding; each FiLM site predicts feature-channel scale and shift.
- **Quantile selector:** pool the measurement to 64×64, apply three stride-2 convolutions (3→16→32→32), global average pooling, and a linear layer producing two values. The lower q lies in (0,0.5), the upper q in (0.5,1). Initial endpoints are 0.05 and 0.95; their separation is not fixed at 0.90.
- **Point prediction:** Q(y, 0.5).
- **Bounds:** native endpoint predictions, anchored around the median and clipped to [0,1]. **No RCPS, fitted scale factors, or calibration images are used for QUTCC inference.**

## Training objective

Each batch accumulates two gradients before one AdamW update:

1. **Pinball fidelity:** 50% of queries are sampled uniformly from (0,1), 25% use the learned lower q, and 25% use the learned upper q. Quantile labels are detached for this term, preventing the selector from reducing the loss just by moving its labels.
2. **Width and coverage:** `0.1 × (mean_width + lambda × soft_miss)` updates both the backbone and the selector. With `gap = relu(LB − target) + relu(target − UB)`, the differentiable miss proxy is `mean(gap / (gap + 1/255))`.

A nonnegative multiplier starts at 1 and follows `lambda = max(0, lambda + 0.005 × (EMA_hard_miss − 0.10))`, using EMA momentum 0.99. Hard miss counts ground-truth RGB values outside the bounds. There is no MSE term in this QUTCC objective. The separate alternating-training helper included as a dependency is **not** the entry point for this experiment.

The objective targets approximately 90% average pixel/channel coverage while encouraging narrow intervals. It does not certify exactly 90% coverage on unseen images or on each image. Evaluate actual coverage and mean interval width together.

## Parameters

| Setting | Value |
|---|---|
| Training / validation / test | 60,000 / 5,000 / 5,000 images per camera |
| Epochs | 50, submitted in two-epoch jobs |
| Batch size | 2 |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0.0001 |
| Gradient clipping | Global norm 1 |
| Precision | BF16 backbone; FP32 quantile selector during interval prediction |
| Seeds | RML: 202610140; DiffuserCam: 202610141 |
| Data order | Independently shuffled each epoch using seed + epoch |
| Checkpointing | Every 1,000 batches, every epoch in latest.pt, milestone weights every 5 epochs |
| Validation | All 5,000 validation images after every epoch |
| QUTCC calibration | None |

See [RML parameters](parameters/rml/config.json), [DiffuserCam parameters](parameters/diffuser/config.json), and [evaluation settings](parameters/evaluation.json). `original_checkpoint_metadata.json` preserves the exact historical cluster metadata: its reserved calibration count and delta belong to the earlier study setup, **not** the current QUTCC inference. Included history and smoke reports are snapshots, not live progress.

These are **hyperparameter/configuration files**, not trained weight tensors. Large checkpoint files and datasets are not embedded in GitHub. Existing cluster checkpoints remain under `~/ConvRML/results/adaptive_q_coverage90_scratch50_20261014/{rml,diffuser}/`.

## Files to read first

| File | Purpose |
|---|---|
| `models/constrained_adaptive_qutcc.py` | Actual joint loss, coverage controller, optimizer step |
| `models/image_adaptive_qutcc.py` | Image-level q queries and native bounds |
| `models/learned_quantile_selector.py` | Small image-dependent q selector |
| `models/convnext_norm_film.py` | FiLM placement and quantile embedding |
| `models/corrected_uncertainty.py` | Quantile reconstruction model |
| `scripts/train_constrained_adaptive50.py` | Active training entry point; resume and checkpoint logic |
| `scripts/export_adaptive_q20_demo.py` | Native-bound inference; no RCPS |

Some shared modules also contain older model variants. They are included to make imports resolve; they are not additional training stages. `cluster_source_manifest.json` records exact original source hashes and environment versions. `package_manifest.json` records this published package and its documented packaging changes.

## Run a new experiment

Use a CUDA-capable environment matching the versions in `cluster_source_manifest.json`, then install `requirements.txt`. Run commands **inside this folder**. The dataset caches are not downloaded by these scripts.

Place or symlink the existing camera caches under:

```
data/optics_intervals_60k_5k_5k_20260928/
data/diffuser_intervals_60k_5k_5k_20260928/
```

Each needs `pairs.npy` and `manifest.json` matching the recorded dataset digest. The loader expects paired uint8 measurement/target RGB images, with shape `[N, 2, 3, H, W]`.

```bash
mkdir -p logs
python scripts/test_constrained_adaptive_qutcc.py
# Run each smoke test on a GPU; check that both smoke.json reports pass.
sbatch scripts/constrained_adaptive50.sbatch smoke --camera rml
sbatch scripts/constrained_adaptive50.sbatch smoke --camera diffuser
# After both smoke jobs succeed, submit independent two-epoch chains to epoch 50.
python scripts/submit_constrained_adaptive50.py
# Optional: gated native-bound demo export after epoch 20.
python scripts/submit_adaptive_q20_demo.py
```

The Slurm wrapper uses the directory from which you submit, rather than a hard-coded project directory. Its environment path and Cornell GPU constraints may need adjustment on another cluster. Do not point a fresh run at the active results directory unless you intend to resume it. `resume_native_qutcc.py` is the audit/recovery utility used for the existing cluster runs; it requires their `latest.pt` checkpoints and is not the fresh-run command.

## Verify and infer

```bash
python -m compileall -q models scripts
python scripts/test_constrained_adaptive_qutcc.py
# Requires the corresponding epoch_020.pt checkpoint:
python scripts/export_adaptive_q20_demo.py --task 0 --epoch 20 --count 50
python scripts/export_adaptive_q20_demo.py --task 1 --epoch 20 --count 50
```

Task 0 is RML; task 1 is DiffuserCam. The exporter uses the first 50 test examples in manifest order and reports their actual coverage, mean width, PSNR and SSIM. Those 50-example aggregates are not full-test results. Baseline im2im, ensemble and MC-dropout RCPS pipelines remain separate and unchanged.
