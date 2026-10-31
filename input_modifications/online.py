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
        exposure_gains = (0.50, 1.50)
        gain_index = torch.randint(
            len(exposure_gains), (), device=image.device, generator=generator
        ).item()
        return change_exposure(image, gain=exposure_gains[gain_index])
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
    """Choose one input condition per image and return (modified_batch, labels).

    Each condition has a 10% chance: clean or one of nine modifications.
    Exposure randomly uses global gain 0.50 or 1.50 within its 10% share.
    Conditions are sampled independently, so batch counts need not be equal.

    Measurements must be float BCHW tensors in [0,1], with C=1 or 3 and
    H/W >= 64. Original measurements are unchanged; targets are not accepted.
    No images are saved. Only JPEG includes codec quantization.

    Reuse a generator on the measurement's device for reproducible random draws.
    Applying this on CPU batches is simplest because JPEG uses a CPU codec.
    Masks/patterns are sampled per image; for frames sharing a sensor, use the
    individual helpers with a persistent mask/pattern instead.
    """
    # 1. Check that every modification supports this batch.
    check_image(measurement)
    batch_size, channels, height, width = measurement.shape
    if channels not in (1, 3) or min(height, width) < 64:
        raise ValueError("The training mixture needs C=1 or 3 and H/W >= 64.")

    # 2. Draw one equally likely condition index for each image.
    condition_indices = torch.randint(
        high=len(CONDITIONS),
        size=(batch_size,),
        device=measurement.device,
        generator=generator,
    ).tolist()

    # 3. Apply only the selected condition to each image.
    modified_images = []
    labels = []
    for image_index, condition_index in enumerate(condition_indices):
        condition = CONDITIONS[condition_index]
        image = measurement[image_index : image_index + 1]  # Keep shape [1,C,H,W].
        modified_image = _apply(image, condition, generator)

        modified_images.append(modified_image)
        labels.append(condition)

    # 4. Join the images back into a batch of the original shape.
    modified_batch = torch.cat(modified_images, dim=0)
    return modified_batch, labels
