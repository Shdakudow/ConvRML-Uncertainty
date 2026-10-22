"""Trainable ConvRML decoder baselines for cached/frozen ConvRML features.

These are head ensembles, not independently trained backbone ensembles.
"""

import copy
import math
from torch import nn
from models.quantile_grid import fresh_head


def mc_head(reconstruction_head, p=0.1):
    if not 0 < p < 1:
        raise ValueError("dropout probability must be in (0,1)")
    head = fresh_head(reconstruction_head)
    layers = []
    for layer in head.f:
        layers.append(layer)
        if isinstance(layer, nn.ReLU):
            layers.append(nn.Dropout2d(p))
    head.f = nn.Sequential(*layers)
    return head


def gaussian_head(reconstruction_head, initial_variance=0.01, seed=0):
    import torch

    if initial_variance <= 1e-6:
        raise ValueError("initial variance must exceed variance floor")
    head = fresh_head(reconstruction_head)
    last = head.f[-1]
    channels = last.out_channels
    out = nn.Conv2d(last.in_channels, channels * 2, 3, padding=1, bias=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():
            out.weight[:channels].copy_(last.weight)
            # Distinct initialization, rather than identical deterministic members.
            out.weight[:channels].add_(torch.randn_like(last.weight) * 1e-3)
            out.weight[channels:].zero_()
            out.bias.zero_()
            if last.bias is not None:
                out.bias[:channels].copy_(last.bias)
            out.bias[channels:].fill_(math.log(math.expm1(initial_variance - 1e-6)))
    head.f[-1] = out
    return head.requires_grad_(True)
