# Fixed corrupted raw measurements · dataset construction and 10-epoch pilot

[Step-by-step workflow](../../docs/RUNNING_EXPERIMENTS.md) · [models/calibration/clipping](../../docs/METHODS_AND_METRICS.md) · [exact source and resume](../../provenance/README.md)

Only measurements are corrupted; ground-truth scenes remain unchanged. `build_dataset.py` creates fixed, deterministic reviewed PNG inputs with hashes and a manifest. `corruptions.py` implements the transformations. The split is **17,999 / 2,000 / 2,000 / 3,000** for training / validation / test / calibration.

The reviewed mixture includes identity inputs and crop/resize, rotation, local rotations, intensity changes, missing pixels, reduced resolution, Gaussian and Poisson perturbations. Speckle, stripe, impulse and scrambling profiles are reserved for separate stress tests. These are sensor-domain stress tests, not a physical optical acquisition model.

After dataset review, each method copied its own **clean epoch-30** checkpoint and adapted for **10 additional epochs**, using fresh Adam moments and constant LR 1e-5. Evaluations are at adaptation epochs 5 and 10. Task 0 is QUTCC; task 1 is Im2im. `submit.py` requires the local `TRAINING_APPROVED` marker after review; it is intentionally not supplied as a blanket approval for new datasets.

The original `protocol.json` includes a submission-status field written before launch; consult the job record or Slurm for actual status. Dataset/review assets remain on Unicorn. [The continuation](../raw_corrupted_continue30_20261019/) extends this pilot to 30 adaptation epochs.

## Files and environment

- `model.py` and `models/`: ConvNeXt/FPN architecture and FiLM implementation.
- `train.py`: training, validation, resumable checkpoints and RNG handling.
- `evaluate.py` and `calibration.py`: separate calibration and held-out test evaluation.
- `data.py`: dataset identities and split checks.
- `protocol.json`: experiment settings; these are **hyperparameters, not trained weights**.
- `run.sbatch` and submission scripts: the actual Cornell Slurm setup.

Use Python 3.11, PyTorch/torchvision, Pillow and the repository-root `requirements.txt`. Run commands from this experiment folder. Data paths default to `$HOME/ConvRML`; `CONVRML_PROJECT` can select another project root. Slurm scripts also contain Cornell-specific executable paths that must be adapted on another system.

Data, split manifests, trained weights, logs and prediction arrays are not included in this source upload. Existing source/checkpoint hashes are intentional resume checks: changing code or dataset locations can invalidate a historical resume. Job JSON files record past submissions; they are not live status and submission scripts refuse duplicate records. Do not blindly rerun scheduling scripts on an active study.

QUTCC samples one uniform scalar quantile per image during training and uses pinball loss only, with 48 FiLM sites. The median prediction (`q=0.5`) is the reconstruction. Calibration searches a **global** lower/upper quantile pair on separate calibration data using the paper's Algorithm 1; it is not a learned image-adaptive selector or the earlier delta-based RCPS implementation. Im2im trains end to end with center MSE and endpoint pinball losses, with separately fitted lower/upper calibration factors. Raw and clipped metrics are retained separately; demo metrics use clipping.
