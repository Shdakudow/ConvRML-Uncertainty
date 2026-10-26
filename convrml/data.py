"""Portable paired-image loaders with the original ordering and normalization.

Both datasets return (2, 3, H, W): measurement first, ground truth second.
Uint8 RGB values are converted to float32 and divided by 255. Neither loader
adds corruption, resizes images, regenerates splits, or changes ground truth.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class NumpyPairs(Dataset):
    """Experiment 2: memory-mapped (N, 2, 3, 256, 256) uint8 cache.

    The manifest lists train/val/test IDs in their existing cache order, or lists a
    separate calibration split. Pass the calibration cache and its own manifest
    when split='calibration'; calibration is never taken from training or test.
    """

    def __init__(self, pairs_path, manifest_path, split):
        manifest = json.loads(Path(manifest_path).read_text())
        roles = [
            role for role in ("train", "val", "test", "calibration") if role in manifest
        ]
        if split not in roles:
            raise ValueError("Requested split is absent from the manifest")
        self.rows = manifest[split]
        self.ids = tuple(str(row["id"]) for row in self.rows)
        self.start = sum(len(manifest[role]) for role in roles[: roles.index(split)])
        self.array = np.load(pairs_path, mmap_mode="r")
        if self.array.ndim != 5 or self.array.shape[1:3] != (2, 3):
            raise ValueError("Expected (N, 2, 3, H, W) measurement/target pairs")
        if self.array.dtype != np.uint8 or len(self.array) != sum(
            len(manifest[r]) for r in roles
        ):
            raise ValueError(
                "Cache dtype/count does not match the uint8 split manifest"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Pair index is outside the selected split")
        pair = np.array(self.array[self.start + index], copy=True)
        return torch.from_numpy(pair).float() / 255


class ImagePairs(Dataset):
    """Experiment 3: paired RGB image paths in manifest['roles'][split].

    Clean and fixed-corruption manifests use the same schema. Paths can be absolute
    or relative to the manifest. Models handle padding; this loader does not resize.
    """

    def __init__(self, manifest_path, split):
        self.manifest_path = Path(manifest_path).resolve()
        self.rows = json.loads(self.manifest_path.read_text())["roles"][split]
        self.ids = tuple(str(row["id"]) for row in self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        pair = []
        for field in ("measurement", "target"):
            path = Path(self.rows[index][field]["path"])
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            pair.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255)
        if pair[0].shape != pair[1].shape:
            raise ValueError("Measurement and target must have matching RGB dimensions")
        return torch.stack(pair)
