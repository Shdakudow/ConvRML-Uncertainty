# ConvRML Uncertainty

Compare how four lensless imaging methods reconstruct an image and estimate **where that reconstruction is uncertain**: im2im, MC dropout, deep ensemble, and quantile-conditioned ConvRML (QUTCC).

The code is organized by experiment. Start with the folder that matches the comparison you want to inspect.

| Folder | What it contains |
| --- | --- |
| **[Original 50-epoch comparison](experiments/baseline_50epochs/)** | The earlier large-data RML and DiffuserCam study: im2im, five-member ensemble, 30-pass MC dropout, and the original final-feature-map quantile conditioner. |
| **[Current adjusted comparison](experiments/adjusted_comparison_50epochs/)** | Corrected pinball-only QUTCC, end-to-end im2im, dropout throughout the network, independent ensemble shuffles, and separate lower/upper calibration scales. The corrected rerun is authorized. |
| **[Exploratory archive](archive/exploratory/)** | Toy experiments, smaller-data studies, CRPS/residual-distribution experiments, earlier pilots, and superseded approaches. |

**Naming clarification:** the large-data comparisons use **60,000 training images and 50 epochs**, not 50,000 training images. Each camera also has 5,000 validation images, 5,000 test images, and 2,048 separate calibration images.

## Current comparison

| Method | What changes? |
| --- | --- |
| **Im2im** | Train lower, center, and upper outputs jointly with the entire backbone; no freezing. |
| **Deep ensemble** | Separate initialization and data-order seeds for each member; restore the shuffle RNG exactly on resume. |
| **MC dropout** | Dropout in every encoder block and decoder refinement stage; tune `0.05`, `0.10`, and `0.20` on validation data. |
| **Q-conditioned ConvRML (QUTCC)** | Pinball-only training with 48 FiLM sites; reconstruction comes from the `q=0.5` prediction. |

**FiLM is only used in QUTCC.** The target is 50 total backbone epochs. The implementation retains two-epoch training chunks and checkpoints every five epochs. **Earlier comparison jobs are stopped. The corrected rerun trains all methods from scratch, using six independent job chains.**

- [Current parameters](experiments/adjusted_comparison_50epochs/configs/parameters.json)
- [Original comparison parameters](experiments/baseline_50epochs/configs/parameters.json)
- [Current setup and execution instructions](experiments/adjusted_comparison_50epochs/README.md#setup-and-running)

The parameter JSON files document the settings. They are not trained weights and are not loaded automatically by the scripts.

## Understanding the comparison

Each model produces a reconstruction, a lower bound, and an upper bound. We compare:

- **Reconstruction quality:** PSNR, SSIM, and absolute error against ground truth.
- **Uncertainty width:** `upper bound - lower bound`.
- **Coverage:** the fraction of ground-truth pixel-channel values inside the interval.
- **Cost:** training and inference time.

We want informative, narrow intervals while maintaining adequate coverage. The RCPS target is 90% coverage in expectation under its assumptions; it does not mean every test image must have exactly 90% coverage.

## Using this repository

Each main experiment folder contains its own required Python modules, requirements file, and execution instructions. Run its commands **from that experiment folder**. Small shared modules are copied into both folders so each source package can be inspected without importing the exploratory archive.

Datasets, split manifests, and trained checkpoints are not included. The corrected trainer rejects historical protocol checkpoints rather than silently resuming them under new objectives. The authorized rerun retrains all ten ensemble members with independent initialization and shuffle seeds.

The stopped October 5 variant is preserved in [a separate archive](archive/superseded_adjusted_20261005/). Historical results and checkpoints remain distinct from the corrected protocol.

## Demo and sources

- [Interactive demo](https://shdakudow.github.io/ConvRML-Uncertainty-Demo/) · [Separate demo repository](https://github.com/Shdakudow/ConvRML-Uncertainty-Demo)
- [Upstream ConvRML](https://github.com/lakabuli/ConvRML) · [Original QUTCC source used here](https://github.com/Shdakudow/code-for-quant) (private)
- [License](LICENSE) · [Upstream citation](CITATION.cff)

The demo represents earlier completed experiments. Updating this code repository does not update or deploy the demo.
