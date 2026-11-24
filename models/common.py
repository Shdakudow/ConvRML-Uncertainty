"""Shared loss functions and predictive interval utilities for the uncertainty models."""

import torch

# Initial AdamW weight decay (step 0); co-scheduled with the learning rate on a cosine decay.
# MCDropout also reads this at inference as the precision term in the Gaussian process variance approximation.
INITIAL_WEIGHT_DECAY = 1e-4

# 95th percentile of N(0,1); symmetric [−z, +z] covers 90% probability mass.
_Z_95 = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.95)))


def load_legacy_backbone_keys(module, state_dict, prefix, *args):
    """Pre-load hook for backward compatibility with checkpoints saved before the attribute was renamed from baseModel to backbone."""
    old_prefix = prefix + "baseModel."
    for key in list(state_dict):
        if key.startswith(old_prefix):
            state_dict[prefix + "backbone." + key[len(old_prefix):]] = state_dict.pop(key)


def pinball_loss(prediction, target, quantile):
    """Pinball (quantile regression) loss. Accepts a scalar quantile or one quantile per image; the value broadcasts over RGB pixels and spatial dimensions."""
    q = torch.as_tensor(quantile, device=prediction.device).reshape(-1, 1, 1, 1)
    error = target - prediction
    return torch.maximum(q * error, (q - 1) * error).mean()


def gaussian_interval_from_mean_variance(mean, variance):
    """Symmetric 90% Gaussian prediction interval: mean ± z₀.₉₅ · √variance, per pixel."""
    radius = _Z_95 * variance.sqrt()
    return mean - radius, mean + radius
