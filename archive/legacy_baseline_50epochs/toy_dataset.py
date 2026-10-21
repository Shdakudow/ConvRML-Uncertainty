"""Synthetic Fourier-blur-and-crop inverse problem built from RML targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from natsort import natsorted
from torch.utils.data import Dataset


class FourierCropDataset(Dataset):
    """Generate y = crop(F⁻¹(F(k)F(x))) + noise from natural RGB targets.

    The unobserved border and randomized PSF/noise make the inverse problem
    deliberately ambiguous.  ``measurement`` is zero outside the sensor crop;
    the target is the full square image x in [0, 1].
    """
    def __init__(self, root, start, count, image_size=64, crop_size=48, seed=20260918, psf="gaussian"):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.seed = int(seed)
        self.psf = psf
        if not 0 < self.crop_size <= self.image_size:
            raise ValueError("crop_size must be in (0, image_size]")
        target_dir = self.root / "gt2rml"
        if not target_dir.is_dir():
            target_dir = self.root / "4x_undistorted_GT2RML"
        files = natsorted(path.name for path in target_dir.iterdir() if path.suffix.lower() in {".tif", ".tiff"})
        self.target_dir = target_dir
        self.files = files[start : start + count]
        if len(self.files) != count:
            raise ValueError(f"Requested {count} source images from offset {start}, found {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def _random_kernel(self, generator):
        size = self.image_size
        coordinates = torch.arange(size, dtype=torch.float32) - size // 2
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        if self.psf == "motion":
            length = torch.empty(()).uniform_(5.0, 17.0, generator=generator)
            angle = torch.empty(()).uniform_(0.0, float(torch.pi), generator=generator)
            along = xx * angle.cos() + yy * angle.sin()
            across = -xx * angle.sin() + yy * angle.cos()
            kernel = torch.exp(-0.5 * (across / 0.5) ** 2) * (along.abs() <= length / 2)
            return kernel / kernel.sum()
        if self.psf != "gaussian":
            raise ValueError(self.psf)
        sigma_x = torch.empty((), dtype=torch.float32).uniform_(1.0, 4.0, generator=generator)
        sigma_y = torch.empty((), dtype=torch.float32).uniform_(1.0, 4.0, generator=generator)
        theta = torch.empty((), dtype=torch.float32).uniform_(0.0, float(torch.pi), generator=generator)
        x_rot = xx * torch.cos(theta) + yy * torch.sin(theta)
        y_rot = -xx * torch.sin(theta) + yy * torch.cos(theta)
        kernel = torch.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
        return kernel / kernel.sum()

    def __getitem__(self, index):
        target = tifffile.imread(self.target_dir / self.files[index])
        if target.shape[-1] == 4:
            target = target[..., :3]
        target = torch.from_numpy(np.ascontiguousarray(target)).permute(2, 0, 1).float()
        if target.max() > 1:
            target = target / 255.0
        target = F.interpolate(target[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False, antialias=True)[0]
        target = target.clamp(0.0, 1.0)

        generator = torch.Generator().manual_seed(self.seed + index)
        kernel_fft = torch.fft.rfft2(torch.fft.ifftshift(self._random_kernel(generator)))
        blurred = torch.fft.irfft2(torch.fft.rfft2(target) * kernel_fft[None], s=(self.image_size, self.image_size)).real
        margin = (self.image_size - self.crop_size) // 2
        measurement = torch.zeros_like(blurred)
        sensor = blurred[:, margin : margin + self.crop_size, margin : margin + self.crop_size]
        noise_sigma = torch.empty((), dtype=torch.float32).uniform_(0.01, 0.05, generator=generator)
        sensor = (sensor + torch.randn(sensor.shape, generator=generator) * noise_sigma).clamp(0.0, 1.0)
        measurement[:, margin : margin + self.crop_size, margin : margin + self.crop_size] = sensor
        return measurement, target, Path(self.files[index]).stem
