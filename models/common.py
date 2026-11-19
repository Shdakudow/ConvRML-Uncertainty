"""Losses and Gaussian intervals shared by the uncertainty models."""

import torch

INITIAL_WEIGHT_DECAY = 1e-4

# 95th percentile of N(0,1); symmetric interval for 90% nominal coverage.
_Z_95 = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.95)))


def load_legacy_backbone_keys(module, state_dict, prefix, *args):
    """Accept checkpoints written before baseModel was renamed to backbone."""
    old_prefix = prefix + "baseModel."
    for key in list(state_dict):
        if key.startswith(old_prefix):
            state_dict[prefix + "backbone." + key[len(old_prefix):]] = state_dict.pop(key)


def pinball_loss(prediction, target, quantile):
    """A scalar or one quantile per image broadcasts over its RGB pixels."""
    q = torch.as_tensor(quantile, device=prediction.device).reshape(-1, 1, 1, 1)
    error = target - prediction
    return torch.maximum(q * error, (q - 1) * error).mean()


def gaussian_interval_from_mean_variance(mean, variance):
    radius = _Z_95 * variance.sqrt()
    return mean - radius, mean + radius
