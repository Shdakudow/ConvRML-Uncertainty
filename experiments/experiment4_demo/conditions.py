"""One catalog for repeatable input changes and the demo's plain-language labels."""

import hashlib
import torch
from input_modifications.exposure import change_exposure
from input_modifications.online import _apply

# Exposure is one training family, displayed at its two possible global gains.
CONDITIONS = (
    ("clean", "Clean reference", "Use the original sensor image.", "y′ = y"),
    (
        "gaussian",
        "Gaussian noise",
        "Add a little random noise.",
        "y′ = clip(y + ε), ε ~ N(0, 0.01²)",
    ),
    (
        "poisson",
        "Poisson noise",
        "Add noise whose size grows with brightness.",
        "y′ = clip(Poisson(1330y) / 1330)",
    ),
    (
        "poisson_gaussian",
        "Poisson + Gaussian",
        "Add both kinds of noise.",
        "y′ = clip(Poisson(1330y) / 1330 + ε), SD(ε) = 0.01",
    ),
    (
        "exposure_050",
        "Half exposure",
        "Make every pixel half as bright.",
        "y′ = clip(0.50y)",
    ),
    (
        "exposure_150",
        "Higher exposure",
        "Make every pixel 1.5 times as bright.",
        "y′ = clip(1.50y)",
    ),
    (
        "dead_pixels",
        "Dead pixels",
        "Set 10% of pixel positions to zero in all RGB channels.",
        "y′ = y(1 − M), exactly 10% of positions masked",
    ),
    (
        "fixed_pattern",
        "Fixed-pattern noise",
        "Give each pixel a small gain and offset.",
        "y′ = clip(gy + b), g = max(0, 1 + N(0, 0.01²)), b ~ N(0, 0.003²)",
    ),
    (
        "gaussian_blur",
        "Gaussian blur",
        "Mix nearby pixels with a small blur filter.",
        "y′ = Gσ * y, σ = 0.5 pixels, filter = 5 × 5",
    ),
    (
        "resolution",
        "Lower resolution",
        "Average 64 × 64 blocks, then resize back.",
        "y′ = crop(resize(avgpool₆₄(pad(y)))); 270 × 480 → 5 × 8 bins",
    ),
    (
        "jpeg",
        "JPEG compression",
        "Compress and decode once at quality 50.",
        "y′ = decode(JPEG₅₀(y)), RGB 4:4:4",
    ),
)
IDS = tuple(row[0] for row in CONDITIONS)


def seed_for(split, identifier, condition):
    """Stable per-image seed, independent of model, batching, and worker order."""
    key = f"exp4-v1/{split}/{identifier}/{condition}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def modify(image, condition, *, split, identifier):
    """Return modified BCHW CPU floats; never accept or change a target."""
    if condition not in IDS or image.device.type != "cpu":
        raise ValueError("Expected a catalog condition and a CPU measurement")
    if condition.startswith("exposure_"):
        return change_exposure(
            image, gain={"exposure_050": 0.5, "exposure_150": 1.5}[condition]
        )
    generator = torch.Generator().manual_seed(seed_for(split, identifier, condition))
    return _apply(image, condition, generator)


def calibration_conditions():
    """Exactly match training family weights on 3,000 calibration scenes."""
    labels = []
    for condition in IDS:
        labels.extend([condition] * (150 if condition.startswith("exposure_") else 300))
    order = torch.randperm(
        len(labels), generator=torch.Generator().manual_seed(202611010)
    )
    return tuple(labels[i] for i in order.tolist())


def catalog():
    return [dict(id=i, title=t, description=d, formula=f) for i, t, d, f in CONDITIONS]
