"""Portable paired-image loaders with the original ordering and normalization.

Both datasets return (2, 3, H, W): measurement first, ground truth second.
Uint8 RGB values are converted to float32 and divided by 255. Neither loader
adds corruption, resizes images, regenerates splits, or changes ground truth.

Example for MIRFLICKR image files:
    from torch.utils.data import DataLoader

    dataset = ImagePairs("split.json", split="train")
    measurement, target = dataset[0]  # Each has shape (3, H, W).
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    measurements, targets = batch[:, 0], batch[:, 1]

The batch has shape (B, 2, 3, H, W). Apply online modifications to measurements
after loading; keep targets unchanged. Use NumpyPairs for a NumPy pair cache.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

SPLIT_ORDER = ("train", "val", "test", "calibration")


class NumpyPairs(Dataset):
    """PLD experiment: memory-mapped (N, 2, 3, 256, 256) uint8 cache.

    The manifest lists train/val/test IDs in their existing cache order, or lists a
    separate calibration split. Pass the calibration cache and its own manifest
    when split='calibration'; calibration is never taken from training or test.
    """

    def __init__(self, pairs_path, manifest_path, split):
        """Select a split from the shared cache using its manifest ordering.

        Args:
            pairs_path: uint8 .npy file shaped (N, 2, 3, H, W).
            manifest_path: JSON mapping split names to ordered rows with IDs.
            split: "train", "val", "test", or "calibration".
        """
        manifest = json.loads(Path(manifest_path).read_text())
        available_splits = [name for name in SPLIT_ORDER if name in manifest]
        if split not in available_splits:
            raise ValueError("Requested split is absent from the manifest")

        self.rows = manifest[split]
        self.ids = tuple(str(row["id"]) for row in self.rows)

        # The cache concatenates splits in SPLIT_ORDER. Locate this split's start.
        split_position = available_splits.index(split)
        preceding_splits = available_splits[:split_position]
        self.start = sum(len(manifest[name]) for name in preceding_splits)

        # Memory mapping reads requested pairs without loading the whole cache.
        self.array = np.load(pairs_path, mmap_mode="r")
        if self.array.ndim != 5 or self.array.shape[1:3] != (2, 3):
            raise ValueError("Expected (N, 2, 3, H, W) measurement/target pairs")

        expected_pair_count = sum(len(manifest[name]) for name in available_splits)
        if self.array.dtype != np.uint8 or len(self.array) != expected_pair_count:
            raise ValueError(
                "Cache dtype/count does not match the uint8 split manifest"
            )

    def __len__(self):
        """Return the number of pairs in the selected split."""
        return len(self.rows)

    def __getitem__(self, index):
        """Return float32 (2,3,H,W) in [0,1], measurement followed by target."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Pair index is outside the selected split")
        cache_index = self.start + index
        # Copy the read-only cache slice before making a PyTorch tensor.
        pixels = np.array(self.array[cache_index], copy=True)
        return torch.from_numpy(pixels).float() / 255


class ImagePairs(Dataset):
    """DLMD experiment: paired RGB image paths in manifest['roles'][split].

    Clean and fixed-corruption manifests use the same schema. Paths can be absolute
    or relative to the manifest. Models handle padding; this loader does not resize.
    These are prepared 8-bit RGB exports, not untouched 12-/16-bit sensor arrays.
    """

    def __init__(self, manifest_path, split):
        """Read the selected split's paths; images are opened when requested.

        Args:
            manifest_path: JSON containing roles[split], a list of rows with
                id, measurement.path, and target.path. Relative image paths
                resolve against this JSON file's directory.
            split: An existing split name, such as "train" or "val".
        """
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        self.rows = manifest["roles"][split]
        self.ids = tuple(str(row["id"]) for row in self.rows)

    def __len__(self):
        """Return the number of pairs in the selected split."""
        return len(self.rows)

    def _load_image(self, image_path):
        """Load one RGB image as float32 [C,H,W], resolving relative paths."""
        path = Path(image_path)
        if not path.is_absolute():
            path = self.manifest_path.parent / path

        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB")).copy()

        # Pillow uses [H,W,C]; the model expects [C,H,W] with values in [0,1].
        channels_first = torch.from_numpy(pixels).permute(2, 0, 1)
        return channels_first.float() / 255

    def __getitem__(self, index):
        """Return [measurement, target] as float32 (2,3,H,W) in [0,1]."""
        row = self.rows[index]
        measurement = self._load_image(row["measurement"]["path"])
        target = self._load_image(row["target"]["path"])

        if measurement.shape != target.shape:
            raise ValueError("Measurement and target must have matching RGB dimensions")

        return torch.stack((measurement, target))
