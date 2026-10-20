import torch


def gaussian_interval_from_mean_variance(mean, variance, alpha=0.1):
    """Gaussian moment interval; not a calibrated coverage guarantee."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    z = torch.distributions.Normal(mean.new_tensor(0.0), mean.new_tensor(1.0)).icdf(
        mean.new_tensor(1 - alpha / 2)
    )
    radius = z * variance.clamp_min(0).sqrt()
    return mean - radius, mean + radius


def aggregate_moments(means, variances):
    if means.shape != variances.shape or means.ndim != 5 or len(means) == 0:
        raise ValueError("Expected matching nonempty member,N,C,H,W tensors")
    mean = means.mean(0)
    aleatoric = variances.mean(0)
    epistemic = means.var(0, unbiased=False)
    return mean, aleatoric + epistemic, aleatoric, epistemic
