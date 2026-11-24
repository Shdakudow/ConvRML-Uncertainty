"""Im2im (image-to-image): three RGB output heads — lower bound, centre, and upper bound — trained jointly with pinball and MSE losses."""

from torch import nn
from torch.nn import functional as F

from .common import pinball_loss


class Im2im(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        output = self.backbone(image)
        return output.reshape(len(image), 3, 3, *output.shape[-2:])

    def center(self, image):
        return self(image)[:, 1]

    def bounds(self, image):
        output = self(image)
        return output[:, 1], output[:, 0], output[:, 2]

    def loss(self, image, target):
        center, lower, upper = self.bounds(image)
        return (
            F.mse_loss(center, target)
            + pinball_loss(lower, target, 0.05)
            + pinball_loss(upper, target, 0.95)
        )
