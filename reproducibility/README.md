# Reproduce the comparisons

The portable commands use the public model definitions and explicit data/checkpoint
paths. No cluster account, scheduler, archived trainer or local home directory is
required. [Commands](../archive/publication_support/docs/RUNNING_EXPERIMENTS.md) cover download, fresh training,
resume and calibrated evaluation.

| Record                                                                     | Contents                                                                                                                   |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [experiment_settings.json](../configs/experiment_settings.json)            | Exact model/profile settings, seeds, split sizes and ordered-ID hashes                                                     |
| [splits/](splits/)                                                         | Existing train/validation/test/calibration IDs in their saved order                                                        |
| [checkpoint_index.json](../archive/checkpoints/checkpoint_index.json)                  | Single index of 280 frozen milestone artifacts with release URLs, size and SHA-256; used by the optional download command |
| [checkpoint_release_receipt.json](checkpoint_release_receipt.json)         | Verified upload receipt for the 40 retained QUTCC/raw artifacts                                                            |
| [environment-tested.json](environment-tested.json)                         | Software used for release verification, with scope/device                                                                  |
| [verification.json](verification.json)                                     | Automated checks and their practical scope                                                                                 |
| [cleanup_verification.json](cleanup_verification.json)                     | Post-v0.1.0 archive/unused-code cleanup checks; original release evidence is retained                                      |
| [paper_calibration_verification.json](paper_calibration_verification.json) | Paper-only calibration change checks and unchanged model/training settings                                                 |
| [model-equivalence.json](model-equivalence.json)                           | Exact original-versus-published CPU model fingerprints                                                                     |
| [shared_models_equivalence.json](shared_models_equivalence.json)           | Exact original-versus-consolidated model fingerprints, losses/gradients and unchanged settings                             |
| [constraints-tested.txt](constraints-tested.txt)                           | Tested numerical dependency versions                                                                                       |
| [DATA.md](DATA.md)                                                         | Original datasets, prepared input formats and access requirements                                                          |

## Dataset splits

| Dataset                |  Train | Validation |  Test | Calibration | Input size  |
| ---------------------- | -----: | ---------: | ----: | ----------: | ----------- |
| RML                    | 60,000 |      5,000 | 5,000 |       2,048 | RGB 256×256 |
| Normalized DiffuserCam | 60,000 |      5,000 | 5,000 |       2,048 | RGB 256×256 |
| Clean raw DiffuserCam  | 17,999 |      2,000 | 2,000 |       3,000 | RGB 270×480 |
| Fixed-corruption raw   | 17,999 |      2,000 | 2,000 |       3,000 | RGB 270×480 |

Calibration IDs are disjoint from all other roles. Hashes verify membership and
order; the prepared-data manifest hash is also recorded at runtime. The current
prototype checks disjoint split IDs and records the supplied manifest; published
counts/hashes are reference records rather than restrictions on new datasets. These are
split/manifest identities, not a claim that every external image byte has a
published content checksum. The loaders preserve target data without generating
new splits, augmentation or resizing.

## Training settings

- Clean QUTCC and raw Im2im: Adam, betas (0.9,0.99), no weight decay, effective
  batch 4; microbatch 2 for square inputs / 1 for raw. Fifty epochs, 30% warmup
  from 1e-5 to 1e-4, then cosine to 1e-9. QUTCC uses only uniform sampled-q pinball.
- parallel lensless Im2im / ensemble / MC: BF16 AdamW, betas (0.9,0.999), LR 1e-4, weight
  decay 1e-4, batch 2, gradient norm limit 1.0, constant LR, fifty epochs.
- Corruption adaptation: each method's clean epoch-30 weights, fresh Adam moments,
  constant LR 1e-5, effective batch 4 / microbatch 1, thirty adaptation epochs.
  Optimizer/RNG state is retained across resumed adaptation segments.
- Ensemble members use independent initialization, training and data-order seeds;
  all see the complete training split. Main comparison uses five members; ten is
  a separate reference. MC uses dropout p=0.05 at 48 sites and thirty test passes.

These are not identical-optimizer ablations. Report optimization and resource
cost differences alongside accuracy. Seed values and shuffle policies are in the
settings file and every portable checkpoint.

## Checkpoint interpretation

Published checkpoints are epochs 5,10,…,50. The 12 corrupted raw adaptation
checkpoints were withdrawn and deleted on 2026-10-30 at the owner's request;
they are excluded from the download index. Clean checkpoints are retained.
[Deletion verification](checkpoint_withdrawal_20261030.json) records the removed
assets and confirms the other assets in that release were unchanged.
Every model/camera/checkpoint is calibrated independently. The default `paper`
evaluation uses RCPS-HB for all baselines and Algorithm 1 for QUTCC. `native`
is an uncalibrated control. Earlier empirical-matching reports remain historical;
their values are not renamed or relabelled as paper calibration results.
Old raw Im2im used a finite-N scale grid; new evaluations use RCPS-HB, as in parallel lensless.
A failed calibration is a recorded gap, not a zero metric.

The portable trainer uses an explicit new checkpoint schema. Saved historical
model dictionaries load strictly after legacy Im2im names are translated; full
historical optimizer/RNG resumes retain their original archived engine and source
identity. The release does not rewrite existing trained checkpoints or reports.
Current Im2im checkpoints use `backbone.*` for every dataset. Direct head
construction changes fresh Im2im head initialization; loaded predictions are
preserved. Earlier equivalence records describe their dated source snapshots.

The [pre-simplification source](../archive/research_prototype_20261102/) retains
source/version enforcement, Slurm automation and historical migration code.
The baseline trainer supports complete-epoch resume. Generalized and specialist
trainers and their launch records have moved to
[ConvRML-Generalized-Specialists](https://github.com/Shdakudow/ConvRML-Generalized-Specialists).
Use original frozen engines for existing deployments.
