# Paper-style ConvNeXt QUTCC and raw Im2im · 50 epochs

**Consolidated implementation:** current QUTCC is now also included in [`../adjusted_comparison_50epochs/qutcc`](../adjusted_comparison_50epochs/qutcc/). Use the parent [four-model entry point](../adjusted_comparison_50epochs/) for new code exploration. This original source location remains preserved for existing cluster jobs/checkpoints and raw-study provenance.

[Step-by-step workflow](../../docs/RUNNING_EXPERIMENTS.md) · [models/calibration/clipping](../../docs/METHODS_AND_METRICS.md) · [exact source and resume](../../provenance/README.md)

This replaces the superseded learned-adaptive-QUTCC experiment. It adapts the original QUTCC quantile embedding/training and calibration procedure to our ConvNeXt/FPN lensless reconstruction model; it does not reproduce the paper's attention U-Net backbone.

| Task index | Dataset | Method | Train / validation / test / calibration |
| --- | --- | --- | --- |
| 0 | RML | QUTCC | 60,000 / 5,000 / 5,000 / 2,048 |
| 1 | Normalized DiffuserCam | QUTCC | 60,000 / 5,000 / 5,000 / 2,048 |
| 2 | Raw DiffuserCam | QUTCC | 17,999 / 2,000 / 2,000 / 3,000 |
| 3 | Raw DiffuserCam | Im2im | 17,999 / 2,000 / 2,000 / 3,000 |

Adam uses betas (0.9, 0.99), no weight decay and effective batch 4. The first 30% of training warms up from 1e-5 to 1e-4; the remainder cosine-decays to 1e-9. Inputs are padded internally and padding is excluded from losses and metrics. Checkpoints and evaluations are retained every five epochs.

`submit.py` preserves the initial two-epoch job schedule for provenance. `reschedule_five_epochs.sh` records the subsequent move to five-epoch jobs under normal QoS; it is a one-time migration script with active-job checks, not a general fresh launcher. `five_epoch_jobs.json` records that migration. Direct entry points (GPU and data required):

```sh
python train.py --task 2 --smoke
python train.py --task 2 --target-epoch 5
python evaluate.py --task 2 --epoch 5
```

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
