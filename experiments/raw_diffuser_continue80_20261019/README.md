# Clean raw DiffuserCam · continuation from epoch 50 to 80

QUTCC and Im2im each resume their own completed epoch-50 checkpoint from [the paper-style study](../qutcc_paper_style_20261018/). Full model, Adam moments, CPU/CUDA and quantile RNG states and counters are retained. Original checkpoints remain unchanged.

The split stays 17,999 training / 2,000 validation / 2,000 test / 3,000 calibration. Learning rate restarts at 1e-5 and cosine-decays to 1e-9 over the additional 30 epochs. This explicit restart is part of the continuation protocol.

Milestones: **55, 60, 65, 70, 75, 80**. Each method has five-epoch training segments and separate calibration/test jobs. Task 0 is QUTCC; task 1 is Im2im. `continuation.py` enforces the parent checkpoint and schedule. `submit.py` schedules a new chain only if there is no existing job record.

```sh
python test_continuation.py
python train.py --task 0 --smoke
python train.py --task 0 --target-epoch 55
python evaluate.py --task 0 --epoch 55
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
