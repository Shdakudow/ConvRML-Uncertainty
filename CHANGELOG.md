# Changes

## Unreleased

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
