# Corrupted raw DiffuserCam · continuation from adaptation epoch 10 to 30

[Step-by-step workflow](../../docs/RUNNING_EXPERIMENTS.md) · [models/calibration/clipping](../../docs/METHODS_AND_METRICS.md) · [exact source and resume](../../provenance/README.md)


**Current status:** QUTCC and Im2im completed all 30 adaptation epochs and their final calibration/test evaluations.
This resumes [the reviewed corruption pilot](../raw_diffuser_corruptions_20261018/) at adaptation epoch 10, retaining model, Adam moments, RNG states and training counters. It uses the **same fixed corrupted PNGs, unchanged targets and split**. No new corruptions are sampled during training.

Learning rate stays **1e-5**. Milestones are adaptation epochs **15, 20, 25, 30**, with five-epoch training jobs and separate calibration/full 2,000-image test evaluation. The final checkpoint has 30 clean pretraining epochs plus 30 corruption-adaptation epochs, not 30 epochs from scratch. Task 0 is QUTCC; task 1 is Im2im.

`continuation.py` verifies the parent checkpoint and optimizer state; `base_data.py` retains the original dataset checks. Parent data, manifests and checkpoints must be restored at the expected paths to run this source snapshot.

```sh
python test_continuation.py
python train.py --task 0 --smoke
python train.py --task 0 --target-epoch 15
python evaluate.py --task 0 --epoch 15
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
