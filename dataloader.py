"""Load paired RGB measurements and targets from the existing split manifest.

The manifest is a JSON file with a "roles" key mapping split names (train, val,
calibration, test) to lists of rows. Each row has "measurement" and "target"
keys whose "path" values are resolved relative to the manifest's directory.

Images retain their original size. Exported images use RGB pixels in [0,1].
Original MIRFLICKR .npy files use the dataset's BGR, orientation, and intensity
corrections instead of the exported-image conversion.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class ImagePairs(Dataset):
    """Dataset of (measurement, target) image pairs for one split of the manifest."""

    def __init__(self, manifest, split):
        self.root = Path(manifest).resolve().parent
        self.rows = json.loads(Path(manifest).read_text())["roles"][split]

    def __len__(self):
        return len(self.rows)

    def read_image(self, name, measurement=False):
        """Load an image from an exported PNG/JPG file or an original MIRFLICKR NumPy array (HWC layout, float32)."""
        path = self.root / name
        if path.suffix.lower() == ".npy":
            # Original MIRFLICKR arrays are stored in floating-point BGR order with rows in reverse vertical order.
            pixels = np.load(path).astype(np.float32)
            pixels = pixels[::-1, :, ::-1]
            if measurement:
                # Normalises for the 0.9× gain applied to measurement arrays during dataset preparation.
                pixels = pixels / 0.9
            pixels = pixels.clip(0, 1)
        else:
            with Image.open(path) as image:
                pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255
        # np.ascontiguousarray is required because step-−1 slicing produces negative strides, which torch.from_numpy does not accept.
        return torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)

    def __getitem__(self, index):
        row = self.rows[index]
        return (
            self.read_image(row["measurement"]["path"], measurement=True),
            self.read_image(row["target"]["path"]),
        )


def get_loaders(manifest):
    """Return DataLoader instances for the training and validation splits of the given manifest, each with batch size 10 and 2 worker processes."""
    train = ImagePairs(manifest, "train")
    validation = ImagePairs(manifest, "val")
    return (
        DataLoader(train, batch_size=10, shuffle=True, num_workers=2),
        DataLoader(validation, batch_size=10, shuffle=False, num_workers=2),
    )
