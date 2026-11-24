"""Deep ensemble: independently trained members each predict per-channel mean and variance, combined via the law of total variance."""

import torch
from torch import nn
from torch.nn import functional as F

from .common import gaussian_interval_from_mean_variance, load_legacy_backbone_keys


class DeepEnsembleMember(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.register_load_state_dict_pre_hook(load_legacy_backbone_keys)

    def forward(self, image):
        mean, raw_variance = self.backbone(image).chunk(2, dim=1)
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
        # Law of total variance: E[Var] + Var[E] = mean within-member variance plus variance of member means.
        variance = (variances + means.square()).mean(dim=0) - mean.square()
        # ReLU clamps values that become marginally negative due to floating-point cancellation in the variance decomposition.
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
