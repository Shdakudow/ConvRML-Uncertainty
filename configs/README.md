# Recorded experiment settings

[experiment_settings.json](experiment_settings.json) contains the preserved model
profiles, optimizer/schedule settings, seeds, split counts and ordered-ID hashes.

| Dataset | Command selector | Models |
| --- | --- | --- |
| Parallel Lensless Dataset (PLD, ConvRML 100k) | `rml`, `diffuser` | Im2im, ensemble, MC dropout, QUTCC |
| DiffuserCam Lensless Mirflickr Dataset (DLMD) | `raw_clean`, `raw_corrupted` | Im2im, QUTCC |

`python train.py --dataset ...` and `python infer.py --dataset ...` select these
profiles automatically. Prepared data paths are command arguments; no cluster
path is built into these settings. The original numeric IDs remain in metadata
for compatibility with saved checkpoints.

The model definitions are shared in [models/](../models/). PLD uses
`dataset.NumpyPairs` at RGB 256×256; DLMD uses `dataset.ImagePairs` at RGB 270×480.
Both return float32 measurement/target pairs after uint8 / 255 conversion.
Model padding/cropping handles the rectangular dimensions. Data loading, image
IDs, train/validation/test/calibration roles and fixed corruptions are preserved.

PLD has 60,000 training / 5,000 validation / 5,000 test / 2,048 calibration pairs
per camera. DLMD has 17,999 / 2,000 / 2,000 / 3,000 pairs. These are our saved
splits, not the upstream repository's default split configuration.

Historical optimization differs: PLD Im2im/ensemble/MC use BF16 AdamW, while
QUTCC and clean DLMD Im2im use FP32 Adam. Those profiles are retained. DLMD
fixed-corruption adaptation starts from its matching clean epoch-30 weights.
Each dataset needs its own weights and calibration even though its model code
is shared. Existing DLMD comparisons have Im2im/QUTCC checkpoints only.

This follows ConvRML's `configs/` organization while retaining our recorded
settings, splits and preprocessing. It does not import upstream YAML defaults
or switch the prepared cache/image loaders to the upstream native-data loader.
