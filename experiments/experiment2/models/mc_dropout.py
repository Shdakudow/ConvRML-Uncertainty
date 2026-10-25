"""Full-network MC dropout for RGB 256 x 256 RML / normalized DiffuserCam.

Input pairs: uint8 NumPy cache divided by 255, shape (N, 3, 256, 256).
Difference from Im2im: one MSE-trained RGB output; uncertainty comes from repeated
stochastic forwards, not endpoint heads. Dropout follows the MLP in all 36 encoder
blocks and GELU in all 12 decoder stages. Default p=0.05, inference uses 30 passes.
The unwrapped state dictionary matches the trained ``mc_005`` checkpoints.
"""

from contextlib import contextmanager
import torch
from torch import nn

from convrml.backbone import ConvNeXtFPN
from convrml.intervals import gaussian_interval_from_mean_variance


@contextmanager
def dropout_inference(model):
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()
        yield
    finally:
        for m, mode in modes.items():
            m.training = mode


class MCDropout(ConvNeXtFPN):
    """Elementwise dropout in all 36 encoder blocks and 12 decoder conv stages."""

    def __init__(self, p=0.05):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError("Expected 0 < dropout probability < 1")
        self.encoder_dropout = nn.ModuleDict()
        self.dropout_locations = []
        self._dropout_handles = []
        for name, block in list(self.encoder.named_modules()):
            if block.__class__.__name__ == "ConvNeXtBlock":
                key = name.replace(".", "__")
                self.encoder_dropout[key] = nn.Dropout(p)
                self._dropout_handles.append(
                    block.mlp.register_forward_hook(self._dropout_hook(key))
                )
                self.dropout_locations.append(
                    "encoder." + name + ".mlp (before residual addition)"
                )
        for name, block in [(f"refine.{i}", b) for i, b in enumerate(self.refine)] + [
            ("up1.1", self.up1[1]),
            ("up2.1", self.up2[1]),
        ]:
            layers = []
            for layer in block.layers:
                layers.append(layer)
                if isinstance(layer, nn.GELU):
                    layers.append(nn.Dropout(p))
                    self.dropout_locations.append(
                        name + ".after_activation_" + str(len(layers) - 1)
                    )
            block.layers = nn.Sequential(*layers)
        if len(self.encoder_dropout) != 36 or len(self.dropout_locations) != 48:
            raise ValueError("Expected 36 encoder and 12 decoder dropout sites")

    def _dropout_hook(self, key):
        def hook(module, inputs, output):
            return self.encoder_dropout[key](output)

        return hook

    def loss(self, y, target):
        return (self(y).float() - target).square().mean()

    @torch.no_grad()
    def bounds(self, y, samples=30, alpha=0.1):
        """Return mean/lower/upper; dropout is enabled only inside this context.

        These are native Gaussian-moment bounds. Calibration is separate.
        """
        if samples < 2:
            raise ValueError("At least two stochastic predictions are required")
        mean = m2 = None
        with dropout_inference(self):
            for i in range(samples):
                sample = self(y).float()
                if mean is None:
                    mean, m2 = sample, torch.zeros_like(sample)
                else:
                    delta = sample - mean
                    mean = mean + delta / (i + 1)
                    m2 = m2 + delta * (sample - mean)
        variance = m2 / samples
        lower, upper = gaussian_interval_from_mean_variance(mean, variance, alpha)
        return mean, lower, upper
