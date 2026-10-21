# Lensless reconstruction and uncertainty

Start with **Experiment 2** for the four-model comparison, or **Experiment 3** for raw DiffuserCam. Each model produces a reconstruction and lower/upper bounds; uncertainty is `upper - lower`.

| Experiment | What to review | Models |
| --- | --- | --- |
| **[2 · RML and normalized DiffuserCam](experiments/experiment2/)** | Paired-data comparison, 50 training epochs, evaluations every five epochs | Im2im, deep ensemble, MC dropout, QUTCC |
| **[3 · Raw and corrupted raw DiffuserCam](experiments/experiment3/)** | Raw image geometry, clean reconstruction, fixed sensor-corruption adaptation | Im2im and QUTCC |

**Review guide:** [model-code differences](docs/MODEL_COMPARISON.md) · [run commands](docs/RUNNING_EXPERIMENTS.md) · [metric definitions](docs/METHODS_AND_METRICS.md).

## What changes between the models?

All use a ConvNeXtV2-Base encoder and convolutional FPN decoder.

| Model | Code change | Loss | How bounds are produced |
| --- | --- | --- | --- |
| **Im2im** | Nine-channel final head: RGB lower, center and upper; whole network trains jointly | Center MSE + two endpoint pinball losses | Direct endpoints, then separate lower/upper calibration scales |
| **Deep ensemble** | Independent networks with RGB mean and variance heads | Gaussian NLL | Aggregate five or ten members' means and variances, then calibrate |
| **MC dropout** | Dropout at 36 encoder and 12 decoder sites | MSE | Mean/variance of 30 stochastic predictions, then calibrate; selected p=0.05 |
| **QUTCC** | Scalar q embedding modulates 48 normalization sites using FiLM | Pinball only, uniform q per image | Query calibrated lower/upper q; reconstruction uses q=0.5 |

Exp2 uses RCPS-HB for Im2im, ensemble and MC dropout. QUTCC uses the paper's Algorithm 1 quantile search. Exp3 uses that quantile search for QUTCC and adjusted empirical asymmetric scale calibration for Im2im. See [calibration differences](docs/MODEL_COMPARISON.md#calibration-differences).

## Dataset splits

| Dataset | Train | Validation | Test | Separate calibration |
| --- | ---: | ---: | ---: | ---: |
| Exp2 · RML | 60,000 | 5,000 | 5,000 | 2,048 |
| Exp2 · normalized DiffuserCam | 60,000 | 5,000 | 5,000 | 2,048 |
| Exp3 · raw / corrupted raw DiffuserCam | 17,999 | 2,000 | 2,000 | 3,000 |

Exp2's normalized DiffuserCam is the Cornell parallel-camera dataset. Exp3 uses the raw DiffuserCam Lensless Mirflickr dataset. The audited raw training set contains 17,999 usable pairs. Fixed corruptions change measurements while preserving targets and split IDs.

## Setup and saved parameters

Use Python 3.11, CUDA-compatible PyTorch/torchvision and [requirements.txt](requirements.txt). Follow the [workflow guide](docs/RUNNING_EXPERIMENTS.md) for data locations, checkpoint loading, training and evaluation commands.

Settings live beside each experiment's source. Weight binaries are separate: [checkpoint releases](https://github.com/Shdakudow/ConvRML-Uncertainty/releases). [Provenance](provenance/) records exact historical source and the repository path migration. Historical cluster data/output paths inside the source remain recorded as trained; the repository moves do not relocate external files.

## Archive and demo

Older baselines, adaptive-q experiments and exploratory work are in **[archive/](archive/)**. The archive index explains their status and maps former locations. Current implementations are linked above.

[Interactive demo](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/) · [demo repository](https://github.com/Shdakudow/ConvRML-Uncertainty-Demo) · [upstream ConvRML](https://github.com/lakabuli/ConvRML) · [QUTCC reference code](https://github.com/Shdakudow/code-for-quant) · [license](LICENSE)
