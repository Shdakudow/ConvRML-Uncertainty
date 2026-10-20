# Corrected ConvRML comparison

**Reading this snapshot:** the normal-camera Im2im, ensemble, and MC models completed 50 epochs. Current QUTCC uses the separate [original-style implementation](../qutcc_paper_style_20261018/). The older conditioned/adaptive-selector paths here are retained for historical state dictionaries and imports. The scheduling and runtime notes below describe that earlier run, not current live job status.

[Current workflow](../../docs/RUNNING_EXPERIMENTS.md) · [calibration and clipping](../../docs/METHODS_AND_METRICS.md) · [source/formatting provenance](../../provenance/README.md). The stopped October-5 variant remains in [the superseded archive](../../archive/superseded_adjusted_20261005/).

[Parameters](configs/parameters.json) · [Model implementations](models/corrected_uncertainty.py) · [Trainer](scripts/corrected_comparison_study.py) · [Evaluator](scripts/corrected_comparison_evaluate.py)

## What changed?

| Method | Corrected implementation |
| --- | --- |
| **Deep ensemble** | Gaussian NLL, with independent model initialization and independent per-epoch data shuffling for each member. The same data split is shared across methods. |
| **Q-conditioned ConvRML (QUTCC)** | Uniformly sampled quantiles and **pinball loss only**. One training forward/backward pass. Reconstruction is the `q=0.5` output; native bounds use `q=0.05` and `q=0.95`. |
| **Im2im** | A 9-channel RGB head produces lower/center/upper images. Train the entire backbone and head jointly with `MSE(center) + pinball(lower, 0.05) + pinball(upper, 0.95)`. No frozen MSE backbone. |
| **MC dropout** | Elementwise dropout in all 36 ConvNeXt encoder residual branches and 12 decoder activation locations. Train with MSE; activate dropout during stochastic inference. |

FiLM remains exclusive to QUTCC, with 36 encoder and 12 decoder conditioning sites. MC dropout uses `nn.Dropout`, not FiLM or residual-path dropping. Its encoder dropout acts on each block's MLP output before residual addition.

## Independent shuffling and resume

Ensemble member `i` uses initialization seed `20261006 + i` and data-generator seed `20281006 + i`. Each epoch reshuffles the training images. Each checkpoint stores the data, quantile, CPU, CUDA, and optimizer states so resuming does not reset or repeat the shuffle sequence.

A protocol identifier rejects the older shared-shuffle, frozen-head, and hybrid-loss checkpoints. The rerun trains all ten ensemble members (`00` through `09`) from scratch, without reusing earlier member weights. MC-dropout candidates deliberately share initialization and data-order seeds to isolate the dropout-rate comparison.

## Separate lower/upper calibration

The evaluator learns distinct scales on the held-out calibration split:

```text
LB = prediction - lambda_lower * lower_distance
UB = prediction + lambda_upper * upper_distance
```

Both empirical and RCPS calibration use separate scales for every evaluated method. With `alpha=0.10`, each tail targets risk `0.05`. RCPS divides the confidence-failure budget `delta=0.05` into `0.025` per tail, giving a joint bound through a union bound. An image is the calibration unit; its loss is its missed RGB fraction. This targets coverage under the calibration assumptions, not exactly 90% coverage on every image.

Native distances from the reconstruction are floored at `1e-6`, consistent with the prior evaluator. Final predictions and bounds are clipped to `[0,1]` for the main reported metrics. The evaluator records both scales and their tail-risk audits.

## Dataset and training settings

RML and DiffuserCam are trained separately, each with 60,000 training, 5,000 validation, 5,000 test, and 2,048 separate calibration images. Images are RGB `256 x 256`. The retained settings are AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size 2, BF16 autocast, and a 50-total-epoch target.

MC dropout candidates are `0.05`, `0.10`, and `0.20`; final selection uses validation SSIM at 100 passes, with 30/60/100-pass trends. Training-time MC validation is deterministic; it is not the final stochastic comparison.

The JSON file documents settings; the scripts do not read it as an executable configuration.

## Setup and checks

Run commands from this directory using the existing Python/PyTorch environment, or install CUDA-compatible PyTorch/torchvision and then:

```bash
python -m pip install -r requirements.txt
python scripts/test_corrected_comparison.py
python scripts/corrected_comparison_study.py --help
python scripts/corrected_comparison_evaluate.py --help
```

Tests cover independent/resumable shuffles, pinball-only QUTCC, median reconstruction, backbone gradients from all im2im outputs, encoder/decoder dropout, inference-mode restoration, and asymmetric RCPS budgets.

The trainer supports `smoke` and `train` actions and allows at most two new epochs per invocation. The submission workflow gates training behind GPU smoke tests and uses twelve independent sequential chains, limiting concurrent training GPU jobs to twelve without a global barrier after every two epochs. All methods can use H100, H200, RTX 6000 Ada, or A6000; Monakhova nodes remain excluded.

Review cluster settings in `scripts/corrected_comparison.sbatch`, then submit from this directory:

```bash
python scripts/submit_corrected_comparison.py --dry-run
python scripts/submit_corrected_comparison.py
```

The second command submits GPU jobs and refuses to overwrite an existing `jobs.json`. It schedules fresh 50-epoch training for ten ensemble members, QUTCC, im2im, and three dropout candidates on each camera; validation selection and calibrated evaluation follow automatically. Evaluation retains six-way concurrency after all twelve training chains finish.

## Data and outputs

Data caches and weights are not included. The trainer expects paired `pairs.npy` arrays and `manifest.json` files at:

```text
data/optics_intervals_60k_5k_5k_20260928/
data/diffuser_intervals_60k_5k_5k_20260928/
data/formal_rml_calibration95/
data/diffuser_intervals_calibration_20260928/
```

New checkpoints and evaluations use `results/corrected_comparison_20261006/`, separate from historical outputs. Older experimental weights have been selected for removal to free storage; historical metrics, reports, source code and the advisor baseline are retained.

The [original large-data baseline](../baseline_50epochs/) and the [demo](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/) show historical variants; they are not results for this corrected implementation.

## Runtime optimization (October 9)

Training retains batch size 2, the same losses, seeds, data splits, learning rate, and 50-epoch target. Host-to-GPU copies are nonblocking. A separate benchmark compares the original runtime with channels-last convolution layout, cuDNN algorithm selection, and fused AdamW using copies of trained weights and their saved optimizer states. It never updates experiment checkpoints.

Run `python scripts/benchmark_corrected_runtime.py` inside a GPU allocation. The trainer enables the faster runtime only when the completed benchmark matches the GPU model and PyTorch version, verifies full-model optimizer resume, including conversion of saved moment-tensor layouts, reports prediction mean absolute difference at most 0.002, and improves step time by more than 10%. Unsupported or unverified hardware keeps the original runtime. The benchmark uses two fixed training images and measures kernel throughput; it is not a full-epoch or accuracy evaluation.

Changes take effect at the next two-epoch job boundary and retain optimizer moments and RNG states. Runtime choice is recorded in every new checkpoint and history row. Different GPU kernels and tensor layouts can introduce floating-point differences; this is not bitwise-identical continuation. Tests: `python -m unittest scripts.test_corrected_runtime scripts.test_corrected_comparison`.

The initial rollout exposed an optimizer-layout incompatibility when resuming NCHW moments into channels-last parameters. The runtime now converts moment strides to match each parameter, and the benchmark must successfully train from actual saved optimizer states before activation. Completed experiment checkpoints were retained. `recover_runtime_jobs.py` repairs failed/cancelled dependency chains without restarting completed epochs.

The final full-optimizer benchmark on an A6000 measured about 3–7% throughput improvement, below the 10% activation threshold; the optimized layout therefore remains disabled for this report. The report is saved under `reports/`. Pending jobs prefer H100/H200/6000 Ada GPUs, with the existing A6000 eligibility retained as a fallback. Preference does not guarantee allocation or a specific runtime.

On October 9, the live training limit was increased from six to twelve public GPUs with user approval. Existing pending jobs were rewired in place; running jobs and completed epochs were retained. Each model remains in one chain, and the tuning barrier waits for all twelve training tails. Monakhova nodes remain excluded. `submit_corrected_comparison.py` now defaults to `--max-gpus 12`; `--max-gpus 6` remains available for smaller future runs. Scheduling tests: `python -m unittest scripts.test_expand_training_lanes`.
