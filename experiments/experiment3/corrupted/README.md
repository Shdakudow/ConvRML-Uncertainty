# Fixed corruptions and adaptation epochs 1–10

[`corruptions.py`](corruptions.py) defines the sensor-domain operations;
[`build_dataset.py`](build_dataset.py) materializes reviewed, deterministic PNGs
and a manifest. Each measurement receives one operation; targets and paired
split IDs stay unchanged. Extra preview stress tests are not all in the training
mixture.

`data.py` reads those existing reviewed files. `train.py` initializes each method
from its own clean epoch-30 weights, with a fresh Adam optimizer at LR 1e-5.
The pilot runs ten adaptation epochs. Task **0 = QUTCC / 1 = Im2im**.

```bash
python build_dataset.py --preview-only
python test_corruptions.py
# GPU, existing approved dataset/checkpoint required:
python train.py --task 0 --target-epoch 5
python evaluate.py --task 0 --epoch 10
```

The preserved trainer requires `TRAINING_APPROVED` after dataset review; no
blanket approval marker is distributed. Continue from epoch 10 using
[corrupted_continuation/](../corrupted_continuation/).

[Setup and source identities](../../../docs/RUNNING_EXPERIMENTS.md#experiment-3).
