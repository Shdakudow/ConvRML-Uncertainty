"""Empirical shared-scale calibration for MC dropout and deep ensembles."""

import torch

from models.common import gaussian_interval_from_mean_variance


def scaled_bounds(center, lower, upper, scale):
    """Scale both distances about the reconstruction, with the reference endpoint floor."""
    lower_limit = center - 1e-6
    upper_limit = center + 1e-6
    lower = center - scale * (center - torch.minimum(lower, lower_limit))
    upper = center + scale * (torch.maximum(upper, upper_limit) - center)
    return torch.minimum(lower, lower_limit), torch.maximum(upper, upper_limit)


class GlobalGridCalibration:
    def __init__(self, model, method):
        self.model = model
        self.method = method
        self.max_scale, self.grid_size = {"mcdropout": (50, 2000), "ensemble": (10, 1000)}[method]
        self.parameters = {}

    def _native_bounds(self, image):
        if self.method == "ensemble":
            return self.model.bounds(image)
        draws = self.model.predict_samples(image)
        variance, center = torch.var_mean(draws, dim=0, unbiased=False)
        lower, upper = gaussian_interval_from_mean_variance(center, variance)
        return center, lower, upper

    @torch.inference_mode()
    def fit(self, loader, device):
        """Count the first covering grid point per pixel using one inference per batch."""
        self.model.eval()
        grid = torch.linspace(0, self.max_scale, self.grid_size, device=device)
        # The last bin counts pixels still missed at the largest scale.
        counts = torch.zeros(self.grid_size + 1, dtype=torch.float64)
        n_images = 0
        for image, target in loader:
            target = target.to(device)
            center, lower, upper = self._native_bounds(image.to(device))
            left = torch.zeros_like(target, dtype=torch.long)
            right = torch.full_like(left, self.grid_size)
            while torch.any(left < right):
                middle = (left + right) // 2
                scale = grid[middle.clamp_max(self.grid_size - 1)]
                lo, hi = scaled_bounds(center, lower, upper, scale)
                covered = (lo <= target) & (target <= hi)
                left = torch.where((left < right) & ~covered, middle + 1, left)
                right = torch.where(covered, middle, right)
            histogram = torch.bincount(left.flatten(), minlength=len(counts))
            counts += histogram.cpu().double() / target[0].numel()
            n_images += len(target)

        # At scale j, all bins above j still miss. Each image has equal weight.
        risks = counts[1:].flip([0]).cumsum(0).flip([0]) / n_images
        target_risk = 0.1 - 0.9 / n_images
        passing = torch.where(risks < target_risk)[0]
        if not len(passing):
            raise ValueError(f"No grid scale up to {self.max_scale} reached risk below {target_risk}.")
        index = int(passing[0])
        self.parameters = dict(
            procedure="Empirical global grid", scale=float(grid[index]),
            grid_size=self.grid_size, max_scale=self.max_scale, n_images=n_images,
            alpha=0.1, target_coverage=0.9, adjusted_risk_target=target_risk,
            calibration_coverage=1 - float(risks[index]),
        )
        return self.parameters

    @torch.inference_mode()
    def predict(self, image):
        center, lower, upper = self._native_bounds(image)
        lower, upper = scaled_bounds(center, lower, upper, self.parameters["scale"])
        return center, lower, upper
