"""Adapted from Downloads/mc_dropout.py for image-only RGB ConvRML models."""

from contextlib import contextmanager
import torch
from torch import nn
from .common import gaussian_interval_from_mean_variance


@contextmanager
def dropout_inference(model):
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()
        yield
    finally:
        for m, mode in modes.items():
            m.training = mode


class MCDropout(nn.Module):
    def __init__(
        self,
        base_model,
        *,
        num_mc_samples=30,
        observation_variance=0.0,
        base_interval_alpha=0.1,
    ):
        super().__init__()
        if num_mc_samples < 2 or observation_variance < 0:
            raise ValueError("Invalid MC configuration")
        self.baseModel = base_model
        self.num_mc_samples = num_mc_samples
        self.observation_variance = observation_variance
        self.base_interval_alpha = base_interval_alpha

    def predict_mean(self, x, *, num_samples=1):
        if self.training and num_samples == 1:
            return self.baseModel(x)
        return self.predict_samples(x, num_samples=num_samples).mean(0)

    @torch.no_grad()
    def predict_samples(self, x, *, num_samples=None):
        n = self.num_mc_samples if num_samples is None else num_samples
        if n < 1:
            raise ValueError("num_samples must be positive")
        with dropout_inference(self.baseModel):
            return torch.stack([self.baseModel(x) for _ in range(n)])

    @torch.no_grad()
    def moments(self, x):
        # Online moments avoid storing T full image predictions.
        with dropout_inference(self.baseModel):
            mean = None
            m2 = None
            for i in range(self.num_mc_samples):
                sample = self.baseModel(x)
                if mean is None:
                    mean = sample.clone()
                    m2 = torch.zeros_like(sample)
                else:
                    delta = sample - mean
                    mean = mean + delta / (i + 1)
                    m2 = m2 + delta * (sample - mean)
        epistemic = m2 / self.num_mc_samples
        return mean, epistemic + self.observation_variance

    def forward(self, x):
        mean, variance = self.moments(x)
        lo, hi = gaussian_interval_from_mean_variance(
            mean, variance, self.base_interval_alpha
        )
        return torch.cat([lo, mean, hi], dim=1)
