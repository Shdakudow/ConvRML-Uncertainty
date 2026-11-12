"""MC dropout: reconstruct repeatedly with dropout active to estimate spread."""

import torch
from torch import nn
from torch.nn import functional as F

from .common import gaussian_interval_from_mean_variance


def add_dropout(module):
    """Add the draft's p=0.1 dropout after encoder and decoder activations."""
    for name, child in list(module._modules.items()):
        if isinstance(child, (nn.GELU, nn.ReLU)):
            module._modules[name] = nn.Sequential(child, nn.Dropout(0.1))
        else:
            add_dropout(child)


class MCDropout(nn.Module):
    def __init__(self, base_model, dataset_size):
        super().__init__()
        self.baseModel = base_model
        self.dataset_size = dataset_size
        add_dropout(self.baseModel)

    def center(self, image):
        return self.baseModel(image)

    def predict_samples(self, image):
        modes = {module: module.training for module in self.baseModel.modules()}
        self.baseModel.eval()
        for module in self.baseModel.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        samples = torch.stack([self.baseModel(image) for _ in range(50)], dim=0)
        for module, mode in modes.items():
            module.training = mode
        return samples

    def bounds(self, image):
        samples = self.predict_samples(image)
        mean = samples.mean(dim=0)
        # Keep the observation-noise term from your draft: length scale 1,
        # dropout 0.1, and weight decay 1e-4. Variance is in squared intensity units.
        inverse_precision = 2 * self.dataset_size * 1e-4 / (1 - 0.1)
        variance = samples.var(dim=0, unbiased=False) + inverse_precision
        lower, upper = gaussian_interval_from_mean_variance(mean, variance)
        return mean, lower, upper

    def forward(self, image):
        mean, lower, upper = self.bounds(image)
        return torch.cat((lower, mean, upper), dim=1)

    def loss(self, image, target):
        return F.mse_loss(self.center(image), target)
