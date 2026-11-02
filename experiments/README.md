# Optional studies and entry points

The portable comparison lives in root `train.py` and `calibration.py`. The studies
here reuse `models/` and `dataset.py`; they do not define another model backbone.
Use each study's protocol rather than assuming all saved runs share settings.

| Study | Start here | Difference from the portable comparison |
| --- | --- | --- |
| [Generalized inputs](generalized_inputs/README.md) | `generalized_inputs/train.py`: `run`, `run_smoke_test`, `train_batch` | CUDA training with online measurement changes, fixed validation panel and mid-epoch preemption recovery |
| [Gaussian specialist](gaussian_specialist/README.md) | `gaussian_specialist/protocol.json`, then `submit.py` | The generalized engine with Gaussian-only training and matched epoch-zero weights |
| [Robustness comparison](experiment4_demo/README.md) | `experiment4_demo/evaluate.py`, then `publish.py` | Frozen clean/generalized checkpoints, a shared calibration mixture, explicitly empirical calibration, and verified web exports |

For generalized training, follow the saved `protocol.json` to
`modify_training_measurements` → `input_modifications/online.py` → `presets.py`.
Only measurements enter these transforms. Validation uses repeatable draws;
calibration and test pairs are not used by the trainer.

`generalized_inputs/slurm.py` builds submission commands and records accepted job
IDs. The submitters require frozen source and preserve dependencies so retries
do not duplicate recorded jobs. Reading these files does not submit anything.

For the robustness comparison, `conditions.py` defines repeatable inputs and
labels, `empirical.py` fits calibration parameters, and `evaluate.py` writes
metrics/images. `publish.py` checks the completed export before publishing the
formatted `web/` template. Its empirical coverage is not an RCPS certificate;
the root calibration workflow retains its separate paper procedures.

Existing cluster jobs run their frozen snapshots. To resume an existing study,
retain its original source/protocol; updated code is for a separately identified
checkout or run. Historical manifests, launch records and weight hashes stay in
`reproducibility/` and the archive.
