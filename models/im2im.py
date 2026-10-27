"""One end-to-end Im2im model for prepared PLD and DLMD RGB measurements.

Inputs: floating RGB measurements (N,3,H,W) in [0,1]; target has the same shape.
Outputs: (N, 3 heads, 3 RGB channels, H, W), ordered lower / center / upper.
Predictions are unbounded normalized intensities; endpoint ordering is not enforced.
Difference from QUTCC: three fixed heads; no quantile input or FiLM modulation.
The checkpoint_prefix option preserves PLD ``base.*`` or DLMD ``net.*`` state
names. It changes serialization names only, not the architecture or objective.
"""

import torch
from torch import nn
from torch.nn import functional as F

from convrml.backbone import ConvNeXtFPN


def pinball(prediction, target, q):
    """Scalar mean loss for matching (N,3,H,W) tensors and q of shape (N,).

    Positive target-minus-prediction error is weighted by q; negative error by
    1-q. Thus the two endpoint heads estimate different conditional quantiles.
    """
    error = target - prediction
    q = q.reshape(-1, 1, 1, 1)
    return torch.maximum(q * error, (q - 1) * error).mean()


class Im2im(nn.Module):
    def __init__(self, *, checkpoint_prefix="base"):
        super().__init__()
        if checkpoint_prefix not in ("base", "net"):
            raise ValueError("Checkpoint prefix must be base or net")
        self.checkpoint_prefix = checkpoint_prefix
        backbone = ConvNeXtFPN()
        old = backbone.head
        self.channels = old.out_channels
        backbone.head = nn.Conv2d(
            old.in_channels,
            3 * self.channels,
            old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        backbone.requires_grad_(True)
        self.add_module(checkpoint_prefix, backbone)

    @property
    def backbone(self):
        return getattr(self, self.checkpoint_prefix)

    def forward(self, y):
        """Predict all three residual images in one pass and add the measurement."""
        h, w = y.shape[-2:]
        padded = F.pad(y, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        features = self.backbone.up2(self.backbone.features(padded))
        residuals = self.backbone.head(features).reshape(
            len(y), 3, self.channels, *padded.shape[-2:]
        )
        return (residuals + padded[:, None])[..., :h, :w]

    def center(self, y):
        return self(y)[:, 1]

    def bounds(self, y):
        """Return (center, lower, upper), each unbounded (N,3,H,W)."""
        outputs = self(y)
        return outputs[:, 1], outputs[:, 0], outputs[:, 2]

    def loss(self, y, target):
        """Train all parameters jointly: center MSE plus endpoint pinball losses."""
        # The center estimates a conditional mean; endpoints estimate the native
        # 90% interval. Equal coefficients are the recorded experiment setting.
        center, lower, upper = self.bounds(y)
        return (
            (center.float() - target).square().mean()
            + pinball(lower.float(), target, target.new_full((len(target),), 0.05))
            + pinball(upper.float(), target, target.new_full((len(target),), 0.95))
        )
