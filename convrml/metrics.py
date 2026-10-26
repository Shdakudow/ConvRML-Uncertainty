"""Per-image metrics using the same intensity and SSIM conventions as the runs."""

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def reconstruction_metrics(prediction, target):
    prediction, target = prediction.float(), target.float()
    mse = (prediction - target).square().mean((1, 2, 3))
    return dict(
        psnr=-10 * mse.clamp_min(1e-12).log10(),
        ssim=structural_similarity_index_measure(
            prediction, target, data_range=1.0, reduction="none"
        ).reshape(-1),
        mae=(prediction - target).abs().mean((1, 2, 3)),
    )


def interval_metrics(prediction, lower, upper, target, *, clipped):
    if clipped:
        prediction, lower, upper = [v.clamp(0, 1) for v in (prediction, lower, upper)]
    result = reconstruction_metrics(prediction, target)
    result.update(
        coverage=((target >= lower) & (target <= upper)).float().mean((1, 2, 3)),
        width=(upper - lower).float().mean((1, 2, 3)),
        crossing=(lower > upper).float().mean((1, 2, 3)),
    )
    return result
