"""End-to-end Im2im for clean / corrupted raw DiffuserCam, RGB 270 x 480.

Pairs come from image-file manifests via ``convrml.data.ImagePairs`` (/ 255).
The measurement is padded to 288 x 480; all output heads crop back to 270 x 480.
The three heads and loss match Experiment 2. This wrapper preserves the raw
checkpoints' ``net.*`` names instead of Experiment 2's ``base.*`` names.
"""

from torch import nn
from torch.nn import functional as F

from convrml.backbone import ConvNeXtFPN
from convrml.losses import pinball


class Im2im(nn.Module):
    def __init__(self):
        super().__init__()
        base = ConvNeXtFPN()
        old = base.head
        base.head = nn.Conv2d(old.in_channels, 9, old.kernel_size, padding=old.padding)
        self.net = base

    def forward(self, y):
        h, w = y.shape[-2:]
        padded = F.pad(y, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        features = self.net.up2(self.net.features(padded))
        return (
            self.net.head(features).reshape(len(y), 3, 3, *padded.shape[-2:])
            + padded[:, None]
        )[..., :h, :w]

    def center(self, y):
        return self(y)[:, 1]

    def bounds(self, y):
        """Return (reconstruction, lower, upper), without clipping/calibration."""
        output = self(y)
        return output[:, 1], output[:, 0], output[:, 2]

    def loss(self, y, target):
        output = self(y)
        return (
            (output[:, 1] - target).square().mean()
            + pinball(output[:, 0], target, target.new_full((len(target),), 0.05))
            + pinball(output[:, 2], target, target.new_full((len(target),), 0.95))
        )
