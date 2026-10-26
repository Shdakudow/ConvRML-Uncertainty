"""End-to-end Im2im for RML / normalized DiffuserCam, RGB 256 x 256.

Inputs: float32 (N, 3, 256, 256) measurements, loaded from uint8 NumPy pairs / 255.
Outputs: (N, 3 heads, 3 RGB channels, 256, 256), ordered lower / center / upper.
Difference from QUTCC: three fixed heads; no quantile input or FiLM modulation.
Historical checkpoint names retain the ``base.*`` prefix.
"""

import torch
from torch import nn

from convrml.backbone import ConvNeXtFPN


def pinball(prediction, target, q):
    """Pixelwise quantile loss used by the fixed lower and upper heads."""
    error = target - prediction
    q = q.reshape(-1, 1, 1, 1)
    return torch.maximum(q * error, (q - 1) * error).mean()


class Im2im(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ConvNeXtFPN()
        old = self.base.head
        self.channels = old.out_channels
        self.base.head = nn.Conv2d(
            old.in_channels,
            3 * self.channels,
            old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        self.base.requires_grad_(True)

    def forward(self, y):
        """Predict all three residual images in one pass and add the measurement."""
        features = self.base.up2(self.base.features(y))
        residuals = self.base.head(features).reshape(
            len(y), 3, self.channels, *y.shape[-2:]
        )
        return residuals + y[:, None]

    def bounds(self, y):
        """Return (reconstruction, lower bound, upper bound); no calibration/clipping."""
        outputs = self(y)
        return outputs[:, 1], outputs[:, 0], outputs[:, 2]

    def loss(self, y, target):
        """Train all parameters jointly: center MSE plus endpoint pinball losses."""
        center, lower, upper = self.bounds(y)
        return (
            (center.float() - target).square().mean()
            + pinball(lower.float(), target, target.new_full((len(target),), 0.05))
            + pinball(upper.float(), target, target.new_full((len(target),), 0.95))
        )
