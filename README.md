# ConvRML: reconstruction and uncertainty

Code for comparing **Im2im, deep ensemble, MC dropout, and quantile-conditioned ConvRML (QUTCC)** on RML and DiffuserCam measurements. Each method returns a reconstruction and lower/upper images. Uncertainty is the interval width, `upper - lower`.

**Start here:** [training → calibration → inference guide](docs/RUNNING_EXPERIMENTS.md) · [model and metric definitions](docs/METHODS_AND_METRICS.md) · [source provenance and historical checkpoints](provenance/README.md).

## Which experiment should I open?

| Folder | What it contains |
| --- | --- |
| **[Current four-model implementation](experiments/adjusted_comparison_50epochs/)** | **Current QUTCC is included in `qutcc/`**, alongside end-to-end Im2im, independently shuffled NLL ensemble members and full-network MC dropout. Use `run_model.py` for all four methods. |
| [Original-style QUTCC/raw source snapshot](experiments/qutcc_paper_style_20261018/) | Preserved source location for the completed RML/normalized QUTCC and raw QUTCC/Im2im jobs; current QUTCC is also integrated into the four-model folder above. |
| [Clean raw continuation](experiments/raw_diffuser_continue80_20261019/) | Full-state continuation after epoch 50. **Stopped at epoch 70 by request**; the folder name records its original 80-epoch plan. |
| [Corrupted raw dataset and pilot](experiments/raw_diffuser_corruptions_20261018/) | Fixed, reviewed measurement corruptions with unchanged ground truth; ten adaptation epochs after 30 clean pretraining epochs. |
| [Corrupted raw continuation](experiments/raw_corrupted_continue30_20261019/) | Continuation to **30 adaptation epochs**, retaining optimizer and RNG states. Both QUTCC and Im2im completed training and calibration/test evaluation. |
| [Every-five-epoch evaluation](experiments/experiment2_uncertainty_analysis_20261019/) | Full 5,000-image LPIPS and pixel-wise width/error Pearson/Spearman metrics at epochs 5, 10, …, 50; checkpoint restoration and curve publication helpers. |
| [Inference exports](experiments/raw_diffuser_demo_20261019/) | Saved-checkpoint image/bound exports, including the audited epoch-50 QUTCC export. |

The [original comparison](experiments/baseline_50epochs/), [older adaptive experiments](new%20experiment%20training%20code/), and [exploratory archive](archive/) are retained for provenance. Do not mix their checkpoints, objectives, or calibration reports with the current experiment.

## Data and environment

| Dataset | Train | Validation | Test | Calibration |
| --- | ---: | ---: | ---: | ---: |
| RML | 60,000 | 5,000 | 5,000 | 2,048 |
| Normalized DiffuserCam | 60,000 | 5,000 | 5,000 | 2,048 |
| Raw / fixed corrupted raw DiffuserCam | 17,999 | 2,000 | 2,000 | 3,000 |

The requested raw training count was 18,000; 17,999 usable paired training images passed the dataset audit. Every split uses ground-truth pairs. Validation selects settings; calibration fits interval parameters; test data evaluates the frozen result.

Use Python 3.11 and CUDA-compatible PyTorch/torchvision. Install [requirements.txt](requirements.txt) after PyTorch. Data, split manifests, and weight binaries are separate from this source repository. Existing archived baseline weights are listed in [GitHub Releases](https://github.com/Shdakudow/ConvRML-Uncertainty/releases); this update does not move or delete weights.

## Calibration is not clipping

Ground truth is normalized to `[0,1]` by dividing stored 8-bit image values by 255. Network outputs are unconstrained: **both uncalibrated and calibrated predictions/bounds can be outside `[0,1]`**. Calibration fits quantile levels or separate endpoint scales on the calibration split. Clipping is an explicit subsequent operation:

```python
clipped = raw.clamp(0, 1)
width = clipped_upper - clipped_lower
coverage = ((target >= clipped_lower) & (target <= clipped_upper)).float().mean()
```

The latest QUTCC/raw evaluators save **both raw and clipped** metrics. The demo uses clipped metrics. Uncalibrated outputs are learned estimates, not arbitrary random values. See [the worked example](docs/METHODS_AND_METRICS.md#normalization-calibration-and-clipping).

## Readable source and reproducibility

Current source is formatted with Black, using four-space indentation and expanded control flow. A byte-for-byte [Unicorn source archive](provenance/unicorn_source_20261020.tar.gz) preserves the source used for historical checkpoints; [the manifest](provenance/unicorn_source_20261020.json) records its hashes. Formatting changes source hashes, so a historical **training resume** must use the exact historical source and data identities. Fresh training can use the formatted source after its own smoke test. Frozen inference loads the saved state dictionary without relaxing the training resume checks.

No training jobs are submitted by cloning the repository or reading these instructions. Submission commands are explicitly identified in [the workflow guide](docs/RUNNING_EXPERIMENTS.md).

## Demo and references

- [Interactive demo](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/) · [separate demo repository](https://github.com/Shdakudow/ConvRML-Uncertainty-Demo)
- [Upstream ConvRML](https://github.com/lakabuli/ConvRML) · [QUTCC reference source used here](https://github.com/Shdakudow/code-for-quant)
- [License](LICENSE) · [upstream citation](CITATION.cff)

This repository update publishes research source and instructions. The website is maintained separately.
