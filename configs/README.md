# Recorded experiment settings

[experiment_settings.json](experiment_settings.json) contains the preserved model
profiles, optimizer/schedule settings, seeds, split counts and ordered-ID hashes.

| Dataset | Command selector | Models |
| --- | --- | --- |
| Parallel Lensless Dataset (PLD, ConvRML 100k) | `rml`, `diffuser` | Im2im, ensemble, MC dropout, QUTCC |
| DiffuserCam Lensless Mirflickr Dataset (DLMD) | `raw_clean`, `raw_corrupted` | Im2im, QUTCC |

`python train_setup.py --dataset ...` and `python calibration.py --dataset ...` select these
profiles automatically. Prepared data paths are command arguments; no cluster
path is built into these settings. The original numeric IDs remain in metadata
for compatibility with saved checkpoints.

## Reading the settings

| Key | Meaning |
| --- | --- |
| `batch_size` | Images contributing to one optimizer update |
| `microbatch` | Images processed in one forward/backward call; gradients accumulate to the batch size |
| `epochs` | Complete passes over training data; also the fixed learning-rate schedule horizon |
| `alpha` | Target miss fraction: 0.10 corresponds to 90% coverage |
| `calibration.delta` | RCPS confidence-failure budget; separate from alpha |
| `seed_policy` | Recorded seeds used for model initialization, training randomness, image shuffling and q sampling |

`protocol.profile(...)` combines these fields into the `run` dictionary used
throughout the code. The profile names `baseline`, `paper` and `adaptation`
identify saved optimizer settings; they are not separate network architectures.

The model definitions are shared in [models/](../models/). PLD uses
`dataset.NumpyPairs` at RGB 256×256; DLMD uses `dataset.ImagePairs` at RGB 270×480.
Both return float32 measurement/target pairs after uint8 / 255 conversion.
Model padding/cropping handles the rectangular dimensions. Data loading, image
IDs, train/validation/test/calibration roles and fixed corruptions are preserved.

PLD has 60,000 training / 5,000 validation / 5,000 test / 2,048 calibration pairs
per camera. DLMD has 17,999 / 2,000 / 2,000 / 3,000 pairs. These are our saved
splits, not the upstream repository's default split configuration.

Historical optimization differs: PLD Im2im/ensemble/MC use BF16 AdamW, while
QUTCC and clean DLMD Im2im use FP32 Adam. New runs now use AdamW for all methods. Existing betas, weight decay, precision
and learning-rate schedules are retained; former Adam profiles have zero weight
decay. Historical checkpoints still record their original optimizer. Resume old
Adam runs with their original source; this trainer rejects a changed run profile. DLMD
fixed-corruption adaptation starts from its matching clean epoch-30 weights.
Each dataset needs its own weights and calibration even though its model code
is shared. Existing DLMD comparisons have Im2im/QUTCC checkpoints only.

This follows ConvRML's `configs/` organization while retaining our recorded
settings, splits and preprocessing. It does not import upstream YAML defaults
or switch the prepared cache/image loaders to the upstream native-data loader.
