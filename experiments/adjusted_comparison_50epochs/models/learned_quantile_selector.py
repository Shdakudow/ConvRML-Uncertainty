"""Input-dependent quantile selection for an already trained QUTCC.

This experimental head predicts one pair per image, not a global grid choice.
Fit on training data, select checkpoints on validation, then independently
calibrate the frozen system. The training loss does not certify coverage.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class ImageQuantileSelector(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(32, 2)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([-math.log(9), math.log(9)]))

    def forward(self, measurement):
        small = F.adaptive_avg_pool2d(measurement, (64, 64))
        probabilities = self.head(self.encoder(small)).sigmoid()
        # Start at .05/.95; learn asymmetric tails without a fixed q gap.
        return 0.5 * probabilities[:, 0], 0.5 + 0.5 * probabilities[:, 1]


class LearnedQuantileIntervals(nn.Module):
    def __init__(self, quantile_model, selector=None):
        super().__init__()
        self.quantile_model = quantile_model.requires_grad_(False).eval()
        self.selector = ImageQuantileSelector() if selector is None else selector

    def train(self, mode=True):
        super().train(mode)
        self.quantile_model.eval()
        return self

    def forward(self, measurement):
        q_lower, q_upper = self.selector(measurement)
        with torch.no_grad():
            median = self.quantile_model(measurement).float().clamp(0, 1)
        # Deliberately retain autograd: frozen weights still transmit d output/d q.
        raw_lower = self.quantile_model(measurement, q_lower).float()
        raw_upper = self.quantile_model(measurement, q_upper).float()
        lower = torch.minimum(raw_lower, median).clamp(0, 1)
        upper = torch.maximum(raw_upper, median).clamp(0, 1)
        return dict(
            prediction=median,
            lower=lower,
            upper=upper,
            q_lower=q_lower,
            q_upper=q_upper,
        )


def interval_score(lower, upper, target, alpha=0.1):
    """Fixed-alpha interval score; q itself is not the loss's target level."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return (
        upper - lower + (2 / alpha) * (F.relu(lower - target) + F.relu(target - upper))
    ).mean()
