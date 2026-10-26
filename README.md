# ConvRML-Uncertainty

A ConvNeXt/FPN comparison of lensless reconstruction and uncertainty:
**Im2im, deep ensemble, MC dropout and QUTCC**.

| Dataset experiment | Models | Measurements |
| --- | --- | --- |
| [Parallel lensless](experiments/parallel_lensless/) | [Four models](experiments/parallel_lensless/models/) | RML / normalized DiffuserCam, RGB 256×256 |
| [Raw DiffuserCam](experiments/raw_diffusercam/) | [Two models](experiments/raw_diffusercam/models/) | Clean / corrupted raw DiffuserCam, RGB 270×480 |

```text
experiments/
  parallel_lensless/models/  Four models: RML and normalized DiffuserCam
  raw_diffusercam/models/    Two models: clean and corrupted raw DiffuserCam
convrml/           Shared layers, data loaders, training, calibration and evaluation
reproducibility/   Settings, ordered split IDs, checkpoint index and tested environment
tests/             Local correctness checks
archive/           Optional guides, plotting, CI configuration and historical code
```

## Install

Linux/macOS, Python 3.9 or newer. Install a suitable PyTorch/torchvision build for
your CPU or CUDA device, then:

```bash
python -m pip install -e . -c reproducibility/constraints-tested.txt
python -m convrml.train --help
python -m convrml.evaluate --help
python -m convrml.download --help
```

[Quick start](archive/publication_support/docs/RUNNING_EXPERIMENTS.md) · [Data formats](convrml/README.md) ·
[Model differences](archive/publication_support/docs/MODEL_COMPARISON.md) · [Reproducibility](reproducibility/README.md)

## Weights and evaluation

```bash
python -m convrml.download --dataset rml --method qutcc --epoch 50 --output weights/rml/qutcc
```

With prepared data at `/path/to/rml`, train or evaluate using:

```bash
python -m convrml.train --dataset rml --method qutcc \
  --manifest /path/to/rml/manifest.json --pairs /path/to/rml/pairs.npy \
  --output runs/rml/qutcc --epochs 50 --device cuda

python -m convrml.evaluate --dataset rml --method qutcc \
  --checkpoints weights/rml/qutcc/exp2__rml__qutcc__epoch_050.pt \
  --manifest /path/to/rml/manifest.json --pairs /path/to/rml/pairs.npy \
  --calibration-manifest /path/to/rml/calibration/manifest.json \
  --calibration-pairs /path/to/rml/calibration/pairs.npy \
  --calibration paper --output outputs/paper/rml/qutcc/epoch_050 --device cuda
```

`--dataset` selects the dataset experiment automatically: `rml` / `diffuser` use
`parallel_lensless`; `raw_clean` / `raw_corrupted` use `raw_diffusercam`.
Historical checkpoint filenames and metadata retain their original IDs.

The [checkpoint index](reproducibility/checkpoints.json) identifies every saved
five-epoch milestone, including five-/ten-member ensemble references, with file
size and SHA-256. Downloads verify both before accepting a file. GitHub CLI
access is needed for private release assets; public assets can download without it.
Prepared image data are external; [data availability](reproducibility/DATA.md)
links the original datasets and specifies the exact input formats and split IDs.

`python -m convrml.evaluate` loads frozen weights, fits on the separate calibration
split and reports held-out SSIM, PSNR, MAE, LPIPS-AlexNet, coverage, width,
crossings and Pearson/Spearman correlations of width with absolute error. It saves
per-image CSV, dataset means/valid counts and optional reconstruction/bound arrays.
LPIPS uses clipped RGB; the other metrics retain separate raw/clipped results.
Pretrained LPIPS weights use the standard Torch cache; `--no-lpips` explicitly
omits that metric for tiny/offline fixtures. Prediction and metric time are separate.
[Calibration](archive/publication_support/docs/COVERAGE90.md) defaults to the paper
procedures: RCPS-HB for Im2im, ensemble and MC dropout; Algorithm 1 calibrated
quantiles for QUTCC. `--calibration native` is an uncalibrated control. Calibration
does not force exact test coverage. Superseded empirical matching is archived.

## Verification and attribution

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

Training and evaluation need no archived code or cluster configuration.
[Release verification](reproducibility/verification.json) records the v0.1.0 checks;
[model equivalence](reproducibility/model-equivalence.json)
records unchanged original model predictions and gradients.
[Cleanup verification](reproducibility/cleanup_verification.json) records the
post-release archive move and unused-code removal checks.

[Citation](CITATION.cff) · [Upstream methods and credit](THIRD_PARTY.md) ·
[License](LICENSE) · [Interactive results](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/)

Original run engines, cluster operations and historical source records remain
in [archive/](archive/). Extended guides, plotting/provenance tools and inactive
GitHub Actions configuration are in [publication support](archive/publication_support/).
Existing trained weights retain their parameter names.
