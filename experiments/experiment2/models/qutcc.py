"""QUTCC on the ConvNeXt/FPN backbone, shared by Experiments 2 and 3.

Experiment 2: float RGB (N, 3, 256, 256), from uint8 NumPy pairs divided by 255.
Experiment 3: RGB (N, 3, 270, 480), from paired image files divided by 255.
Inputs are replicate-padded to multiples of 32; outputs are cropped to input size.
One scalar q per image conditions 36 encoder and 12 decoder sites through FiLM.
Only sampled-q pinball is optimized; there is no additional MSE or q selector.
"""

import torch
from torch import nn
from torch.nn import functional as F

from convrml.backbone import ConvNeXtFPN
from convrml.film import FiLMBackbone


def pinball(prediction, target, q):
    """Pixelwise quantile loss, averaged over the batch, RGB channels and pixels."""
    error = target - prediction
    q = q.reshape(-1, 1, 1, 1)
    return torch.maximum(q * error, (q - 1) * error).mean()


class QUTCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FiLMBackbone(ConvNeXtFPN())

    def forward(self, y, q=None):
        """Return the RGB quantile image Q(q | y); default q=0.5 is reconstruction."""
        h, w = y.shape[-2:]
        padded = F.pad(y, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        if q is None:
            q = y.new_full((len(y),), 0.5)
        return self.net(padded, q)[..., :h, :w]

    def center(self, y):
        return self(y)

    def bounds(self, y, lower_q=0.05, upper_q=0.95):
        """Query native bounds; use separately calibrated q values when available."""
        if not 0 <= lower_q <= upper_q <= 1:
            raise ValueError("Expected ordered q values in [0, 1]")
        return (
            self(y),
            self(y, y.new_full((len(y),), lower_q)),
            self(y, y.new_full((len(y),), upper_q)),
        )

    def loss(self, y, target, *, generator=None):
        """Sample q ~ Uniform(0,1) independently per image and minimize pinball."""
        q = torch.rand(
            len(y), device=y.device, dtype=torch.float32, generator=generator
        )
        return pinball(self(y, q), target, q)
