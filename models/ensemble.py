"""Deep ensemble: independently initialized RGB mean and variance predictors."""

import torch
from torch import nn
from torch.nn import functional as F

from .common import gaussian_interval_from_mean_variance


class DeepEnsembleMember(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.baseModel = base_model

    def forward(self, image):
        mean, raw_variance = self.baseModel(image).chunk(2, dim=1)
        variance = F.softplus(raw_variance) + 1e-6
        return mean, variance

    def loss(self, image, target):
        mean, variance = self(image)
        return F.gaussian_nll_loss(mean, target, variance)


class DeepEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def moments(self, image):
        means, variances = zip(*(member(image) for member in self.members))
        means, variances = torch.stack(means), torch.stack(variances)
        mean = means.mean(dim=0)
        variance = (variances + means.square()).mean(dim=0) - mean.square()
        return mean, F.relu(variance)

    def center(self, image):
        return torch.stack([member(image)[0] for member in self.members]).mean(0)

    def bounds(self, image):
        mean, variance = self.moments(image)
        lower, upper = gaussian_interval_from_mean_variance(mean, variance)
        return mean, lower, upper

    def forward(self, image):
        mean, lower, upper = self.bounds(image)
        return torch.cat((lower, mean, upper), dim=1)

    def loss(self, image, target):
        return torch.stack(
            [member.loss(image, target) for member in self.members]
        ).mean()
