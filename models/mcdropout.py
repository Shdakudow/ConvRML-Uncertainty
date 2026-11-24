"""Monte Carlo dropout: draws stochastic samples at inference to estimate predictive uncertainty."""

import torch
from torch import nn
from torch.nn import functional as F

from .common import INITIAL_WEIGHT_DECAY, gaussian_interval_from_mean_variance, load_legacy_backbone_keys

DROPOUT_RATE = 0.1
MC_PASSES = 50


def add_dropout(module):
    for name, child in list(module._modules.items()):
        if isinstance(child, (nn.GELU, nn.ReLU)):
            module._modules[name] = nn.Sequential(child, nn.Dropout(DROPOUT_RATE))
        else:
            add_dropout(child)


class MCDropout(nn.Module):
    def __init__(self, base_model, dataset_size):
        super().__init__()
        self.backbone = base_model
        self.register_load_state_dict_pre_hook(load_legacy_backbone_keys)
        self.dataset_size = dataset_size
        add_dropout(self.backbone)

    def center(self, image):
        return self.predict_samples(image).mean(dim=0)

    def predict_samples(self, image):
        # Sets all modules to evaluation mode to freeze BatchNorm running statistics, then
        # re-enables Dropout layers to retain stochastic sampling. All training modes are restored after sampling.
        modes = {module: module.training for module in self.backbone.modules()}
        self.backbone.eval()
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        samples = torch.stack([self.backbone(image) for _ in range(MC_PASSES)], dim=0)
        for module, mode in modes.items():
            module.training = mode
        return samples

    def bounds(self, image):
        samples = self.predict_samples(image)
        mean = samples.mean(dim=0)
        # GP variance approximation (Gal & Ghahramani, 2016):
        # τ⁻¹ = 2N·λ/(1−p), where N = dataset size, λ = weight decay, p = dropout rate.
        inverse_precision = 2 * self.dataset_size * INITIAL_WEIGHT_DECAY / (1 - DROPOUT_RATE)
        variance = samples.var(dim=0, unbiased=False) + inverse_precision
        lower, upper = gaussian_interval_from_mean_variance(mean, variance)
        return mean, lower, upper

    def forward(self, image):
        mean, lower, upper = self.bounds(image)
        return torch.cat((lower, mean, upper), dim=1)

    def loss(self, image, target):
        return F.mse_loss(self.backbone(image), target)
