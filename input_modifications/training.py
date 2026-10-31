"""Generate one of ten equally likely input conditions without saving images."""

from typing import List, Optional, Tuple

import torch

from ._validation import check_image
from .dead_pixels import apply_dead_pixels, sample_dead_pixel_mask
from .exposure import change_exposure
from .fixed_pattern import apply_fixed_pattern, sample_fixed_pattern
from .gaussian import gaussian_noise
from .gaussian_blur import gaussian_sensor_blur
from .jpeg import jpeg_compression
from .poisson import poisson_noise
from .poisson_gaussian import poisson_gaussian_noise
from .resolution import reduce_sensor_resolution


# Each family, including clean, has probability 1/10 for every image.
CONDITIONS = (
    "clean",
    "gaussian",
    "poisson",
    "poisson_gaussian",
    "exposure",
    "dead_pixels",
    "fixed_pattern",
    "gaussian_blur",
    "resolution",
    "jpeg",
)


def _apply(image, condition, generator):
    """Apply the current experiment setting to one BCHW measurement."""
    if condition == "clean":
        return image.clone()
    if condition == "gaussian":
        return gaussian_noise(image, sigma=0.01, generator=generator)
    if condition == "poisson":
        return poisson_noise(image, peak_photons=1330, generator=generator)
    if condition == "poisson_gaussian":
        return poisson_gaussian_noise(
            image, gaussian_sigma=0.01, peak_photons=1330, generator=generator
        )
    if condition == "exposure":
        brighter = torch.randint(2, (), device=image.device, generator=generator).item()
        return change_exposure(image, gain=1.50 if brighter else 0.50)
    if condition == "dead_pixels":
        mask = sample_dead_pixel_mask(image, drop_rate=0.10, generator=generator)
        return apply_dead_pixels(image, mask)
    if condition == "fixed_pattern":
        pattern = sample_fixed_pattern(
            image, gain_std=0.01, offset_std=0.003, generator=generator
        )
        return apply_fixed_pattern(image, pattern)
    if condition == "gaussian_blur":
        return gaussian_sensor_blur(image, sigma=0.5, radius=2)
    if condition == "resolution":
        return reduce_sensor_resolution(image, factor=64)
    if condition == "jpeg":
        return jpeg_compression(image, quality=50, subsampling=0)
    raise ValueError(f"Unknown input condition: {condition}")


def sample_training_modification(
    measurement: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Return modified measurements and one family label per image.

    Input: float BCHW measurements in [0,1], C=1 or 3, H/W >= 64.
    Sample independently per image: 10% clean, 10% for each of nine methods.
    Exposure's share splits into 5% darker and 5% brighter globally. These are
    sampling probabilities, not modification strengths. Only mixed Poisson +
    Gaussian deliberately combines noise processes; other methods are separate.

    Targets are not accepted or changed. No files are written. Tensor results
    retain floating point precision; JPEG alone includes its codec quantization.
    The generator must match the measurement device. Reuse it between calls to
    obtain new draws; recreate it with the same seed to replay the sequence.

    Dead masks and fixed patterns are sampled per image as independent simulated
    sensors. For frames/clips from one sensor, use the individual method helpers
    with a shared mask/pattern instead. JPEG runs its codec on the CPU, so applying
    this sampler before transferring a training batch to the GPU is simplest.
    """
    check_image(measurement)
    if measurement.shape[1] not in (1, 3) or min(measurement.shape[-2:]) < 64:
        raise ValueError("The training mixture needs C=1 or 3 and H/W >= 64.")
    choices = torch.randint(
        len(CONDITIONS),
        (measurement.shape[0],),
        device=measurement.device,
        generator=generator,
    ).tolist()
    labels = [CONDITIONS[index] for index in choices]
    images = [
        _apply(image.unsqueeze(0), condition, generator)
        for image, condition in zip(measurement, labels)
    ]
    return torch.cat(images, dim=0), labels
