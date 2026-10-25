"""Sensor-coordinate corruptions only. No operation touches the scene target.

A scalar pixel has no orientation: patch_rotation rotates local neighborhoods.
Scrambling is a deliberately destructive, evaluation-only negative control.
"""

import hashlib, math
import torch
from torch.nn import functional as F

TRAIN_CONDITIONS = (
    "crop_resize",
    "rotation",
    "patch_rotation",
    "intensity",
    "missing",
    "low_resolution",
    "gaussian",
    "poisson",
)
OOD_CONDITIONS = ("speckle", "stripe", "impulse", "scramble")
ALL_CONDITIONS = ("clean",) + TRAIN_CONDITIONS + OOD_CONDITIONS
SEVERITIES = ("mild", "moderate", "severe")


def seed_for(role, identifier, epoch=0):
    return int.from_bytes(
        hashlib.sha256(f"raw-corrupt-v1|{role}|{identifier}|{epoch}".encode()).digest()[
            :8
        ],
        "little",
    ) % (2**63 - 1)


def corrupt(y, condition="mixed", severity="moderate", seed=0, return_condition=False):
    assert y.ndim == 3 and y.shape[0] == 3 and y.device.type == "cpu"
    assert severity in SEVERITIES
    g = torch.Generator().manual_seed(seed)
    s = SEVERITIES.index(severity)
    _, h, w = y.shape

    def uniform(low, high):
        return low + (high - low) * float(torch.rand((), generator=g))

    if condition == "mixed":
        # Each image receives one operation; 20% identity avoids forgetting.
        if float(torch.rand((), generator=g)) < 0.2:
            condition = "clean"
        else:
            condition = TRAIN_CONDITIONS[
                int(torch.randint(len(TRAIN_CONDITIONS), (), generator=g))
            ]
    assert condition in ALL_CONDITIONS, condition
    z = y.clone()
    if condition == "crop_resize":
        keep = (0.95, 0.85, 0.65)[s]
        ch, cw = max(1, round(h * keep)), max(1, round(w * keep))
        top = int(torch.randint(h - ch + 1, (), generator=g))
        left = int(torch.randint(w - cw + 1, (), generator=g))
        z = F.interpolate(
            z[:, top : top + ch, left : left + cw][None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]
    elif condition == "rotation":
        angle = math.radians(uniform(-1, 1) * (3, 10, 30)[s])
        c, t = math.cos(angle), math.sin(angle)
        theta = z.new_tensor([[c, -t * h / w, 0], [t * w / h, c, 0]])[None]
        grid = F.affine_grid(theta, (1, 3, h, w), align_corners=False)
        z = F.grid_sample(
            z[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[0]
    elif condition == "patch_rotation":
        side = 32
        prob = (0.05, 0.15, 0.4)[s]
        for i in range(0, h - side + 1, side):
            for j in range(0, w - side + 1, side):
                if float(torch.rand((), generator=g)) < prob:
                    k = int(torch.randint(1, 4, (), generator=g))
                    z[:, i : i + side, j : j + side] = torch.rot90(
                        y[:, i : i + side, j : j + side], k, (-2, -1)
                    )
    elif condition == "intensity":
        strength = (0.1, 0.3, 0.6)[s]
        field = 1 + strength * (2 * torch.rand((1, 3, 6, 10), generator=g) - 1)
        field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=False)[
            0
        ]
        z = z * field
    elif condition == "missing":
        mask = torch.rand((1, h, w), generator=g) < (0.05, 0.2, 0.5)[s]
        z = z.masked_fill(mask, 0)
    elif condition == "low_resolution":
        factor = (2, 4, 8)[s]
        small = F.interpolate(
            z[None], size=(max(1, h // factor), max(1, w // factor)), mode="area"
        )
        z = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)[0]
    elif condition == "gaussian":
        z = z + (0.003, 0.01, 0.03)[s] * torch.randn(z.shape, generator=g)
    elif condition == "poisson":
        photons = (1000, 200, 50)[s]
        z = torch.poisson(z * photons, generator=g) / photons
    elif condition == "speckle":
        z = z * (1 + (0.1, 0.3, 0.6)[s] * torch.randn(z.shape, generator=g))
    elif condition == "stripe":
        z = z + (0.003, 0.01, 0.03)[s] * torch.randn((3, h, 1), generator=g)
    elif condition == "impulse":
        mask = torch.rand((1, h, w), generator=g)
        p = (0.005, 0.02, 0.08)[s]
        z = torch.where(
            mask < p / 2,
            torch.zeros_like(z),
            torch.where(mask < p, torch.ones_like(z), z),
        )
    elif condition == "scramble":
        n = round(h * w * (0.05, 0.2, 1.0)[s])
        index = torch.randperm(h * w, generator=g)[:n]
        source = index[torch.randperm(n, generator=g)]
        z = z.flatten(1)
        z[:, index] = y.flatten(1)[:, source]
        z = z.reshape_as(y)
    result = z.clamp(0, 1).contiguous()
    return (result, condition) if return_condition else result
