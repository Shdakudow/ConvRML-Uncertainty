# Local ConvRML rebuild

This workspace contains the official ConvRML implementation plus a local training
configuration and two controlled capacity knobs.  The default model is structurally
identical to the published ConvNeXt-tiny reconstruction model:

1. A 4-stage ConvNeXt encoder with depths `[3, 3, 9, 3]` and dimensions
   `[96, 192, 384, 768]`.
2. Bilinear upsampling of all four feature maps to `300 x 480`, followed by channel
   concatenation (1,440 channels).
3. A convolutional decoder with widths `[512, 256, 128, 64, 3]`.

The default configuration is
[`configs/PLD_RML/config_local_tiny_tuned.yaml`](configs/PLD_RML/config_local_tiny_tuned.yaml).
Before training, set `CONVRML_DATA_PATH` to the PLD root containing `4x_rml/`
and `4x_undistorted_GT2RML/`; the configuration reads the path from that
environment variable and does not embed cluster-specific paths.

The Unicorn 100k copy at `/share/monakhova/shamus_data/parallel_100k` uses the
equivalent names `rml/` and `gt2rml/`.  The local loader accepts both layouts.

Create the training environment on the training machine:

```bash
python3.11 -m venv .venv
.venv/bin/pip install torch torchvision
.venv/bin/pip install -r requirements.txt
```

The upstream `tifffile==2025.3.13` requirement was unavailable from the local
package index, so this build uses `tifffile==2024.8.30`; this does not alter the
model architecture or TIFF loading interface.

Validate the architecture before accessing the dataset:

```bash
.venv/bin/python scripts/verify_model.py
```

Train and evaluate:

```bash
export CONVRML_DATA_PATH=/absolute/path/to/PLD
.venv/bin/python train.py --config configs/PLD_RML/config_local_tiny_tuned.yaml
.venv/bin/python infer.py --config configs/PLD_RML/config_local_tiny_tuned.yaml
```

For a controlled nearby experiment, use `lr: 0.0003` (already selected) and either
`drop_path_rate: 0.1` or `decoder_channels: [448, 224, 112, 56]`; change one at a
time so the result remains comparable to the published default.

## Unicorn

The project is staged at `~/ConvRML` on Unicorn.  Its shared 100k PLD root is
`/share/monakhova/shamus_data/parallel_100k`, already selected by
`scripts/train_unicorn.sbatch`.  Create the environment once in a GPU allocation:

```bash
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda create -y --prefix ~/.conda/envs/convrml python=3.11
conda activate ~/.conda/envs/convrml
cd ~/ConvRML
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Submit training from `~/ConvRML` with `sbatch scripts/train_unicorn.sbatch` and
monitor it with `squeue --me`.  The batch script requests an AVX2-capable GPU
node with at least 20 GB of GPU memory, which avoids incompatible older nodes.
