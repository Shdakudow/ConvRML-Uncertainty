"""Sensor-plane averaging and resampling with explicit treatment of edge pixels."""

import torch
import torch.nn.functional as F

from ._validation import check_image


def reduce_sensor_resolution(y: torch.Tensor, factor: int = 4) -> torch.Tensor:
    """Average factor-square bins, restore the padded size, and crop to the input.

    Replicate bottom/right edges to a multiple of factor before pooling. For a
    270x480 input and factor=4, pool 272x480 to 68x120, interpolate to 272x480,
    then crop to 270x480. No original rows are discarded or geometrically stretched
    to compensate for a truncated pooling grid. Outputs remain normalized averages.
    """
    check_image(y)
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer.")
    height, width = y.shape[-2:]
    if factor > min(height, width):
        raise ValueError("factor must not exceed image height or width.")
    if factor == 1:
        return y.clone()
    work = y.float() if y.dtype in (torch.float16, torch.bfloat16) else y
    padded = F.pad(
        work, (0, (-width) % factor, 0, (-height) % factor), mode="replicate"
    )
    low_res = F.avg_pool2d(padded, kernel_size=factor, stride=factor)
    restored = F.interpolate(
        low_res, size=padded.shape[-2:], mode="bilinear", align_corners=False
    )
    return restored[..., :height, :width].clamp(0, 1).to(y.dtype)
