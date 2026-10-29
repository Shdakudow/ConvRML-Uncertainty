# Changes

## Unreleased

- Standardize Im2im names to `backbone.*` and translate legacy `base.*`/`net.*`
  weights in the checkpoint loader, independently of dataset selection. Construct
  three RGB heads directly rather than creating and replacing a default head.
  Loaded predictions are preserved; fresh Im2im head initialization changes because
  discarded random draws are removed. Historical training snapshots are untouched.
- Document image dimensions as dataset settings, and add automatic padding/cropping
  for MC dropout and Gaussian ensemble members. All four default models now accept
  rectangular/non-stride-aligned RGB images. Existing 256x256 behavior, state keys
  and recorded benchmark profiles remain compatible.
- Explain the backbone feature sizes, internal normalization, q-embedding names,
  validation versus test prediction, settings terminology and saved output files.
  Expand CLI help and function documentation without changing numerical code.
- Name lower/upper calibration states explicitly; document tensor shapes,
  intensity units and scientific choices at model/data/loss/metric boundaries.
  Record paper-to-code adaptations, including asymmetric RCPS and MC-dropout's
  model-spread-only variance, without changing trained model behavior.
- Make `train.py` and `infer.py` the readable workflow implementations, with named
  epoch, calibration, evaluation and export steps. Group checkpoint save/resume
  logic in `convrml/checkpoints.py` and calibration fitting in `calibration.py`.
  Keep the older package commands as thin compatibility entry points.
- Use explicit ConvNeXt block types, descriptive FiLM/dropout state names and
  named embedding dimensions. Numerical behavior and serialized parameters stay
  unchanged; add a short code-reading guide to the README.
- Define the four methods once in `models/`, with shared adaptive-size Im2im/QUTCC
  and legacy checkpoint-name compatibility. Put data loading in `dataset.py`,
  recorded settings in `configs/`, and add common `train.py`/`infer.py` entry points.
  Label datasets as PLD (ConvRML 100k) and DLMD (DiffuserCam Lensless Mirflickr).
- Put pinball formulas directly in the Im2im/QUTCC model files and remove
  `convrml/losses.py`; objectives, gradients and checkpoint parameters are unchanged.
- Add evaluation LPIPS-AlexNet and per-image Pearson/Spearman uncertainty/error
  correlations, valid counts and five-epoch quality curves. Separate prediction
  timing from metric computation; omit region correlations and preserve models/losses.
- Default to paper calibration only: RCPS-HB for all baselines, including raw
  Im2im, and Algorithm 1 calibrated quantiles for QUTCC. Archive empirical matching
  and the old combined reference evaluator; retain native intervals as a control.
- Move optional guides, plotting/provenance tools and GitHub Actions configuration
  to `archive/publication_support/`; retain short runnable commands in the README.
- Remove unused FiLM feature/head accessors, an unused import and the unused
  ensemble mean alias. Model computations and state-dictionary names are unchanged.
- Keep active runtime, calibration and checkpoint validation modules; they are
  required by training and evaluation. Retain the existing v0.1.0 release snapshot.
- Existing frozen weights load unchanged. Resume a portable run started before
  this cleanup from its original checkout: source-identity guards remain strict.

## 0.1.0

- Four annotated Exp2 models and two Exp3 models, with common layers implemented once.
- Portable training, epoch-boundary resume, calibration and held-out evaluation.
- Original model parameter names retained; existing weights remain compatible.
- Explicit empirical, reference and native calibration modes, with separate raw/clipped metrics.
- Ordered split IDs, settings, seed policies, verified checkpoint downloads and environment records.
- Cluster/job/demo tooling separated under archive; no cluster dependency in the public workflow.
- Software citation separated from upstream paper attribution.
