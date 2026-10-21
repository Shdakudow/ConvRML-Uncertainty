"""Quantile-only FiLM after every ConvNeXt block and decoder GELU.

Hooks leave checkpoint keys unchanged. They are active only during a quantile
forward; the ordinary reconstruction always uses the unconditioned backbone.
This model is intended for sequential forwards, without activation checkpointing
or concurrent forwards on the same instance.
"""

from contextlib import contextmanager
import torch
from torch import nn
from models.formal_quantile import QuantileRML


class IdentityFiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.affine = nn.Linear(64, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x, embedding):
        scale, shift = self.affine(embedding).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class EverywhereQuantileRML(QuantileRML):
    def __init__(self, base, levels=(0.05, 0.95)):
        super().__init__(base, "conditioned", levels=levels)
        self.deep_embedding = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.deep_films = nn.ModuleDict()
        self.film_locations = []
        self._embedding = None
        self._handles = []
        for name, module in list(base.named_modules()):
            channels = None
            if module.__class__.__name__ == "ConvNeXtBlock":
                channels = module.conv_dw.in_channels
            elif isinstance(module, nn.GELU) and name.startswith(
                ("refine.", "up1.", "up2.")
            ):
                channels = 256
            if channels is not None:
                key = name.replace(".", "__")
                self.deep_films[key] = IdentityFiLM(channels)
                self.film_locations.append({"module": name, "channels": channels})
                self._handles.append(module.register_forward_hook(self._hook(key)))
        if not self.deep_films:
            raise ValueError("No FiLM insertion sites found")

    def _hook(self, key):
        def apply(module, inputs, output):
            if self._embedding is None:
                return output
            return self.deep_films[key](output, self._embedding)

        return apply

    @contextmanager
    def condition(self, q):
        if self._embedding is not None:
            raise RuntimeError("Nested/concurrent quantile forward is unsupported")
        self._embedding = self.deep_embedding(q.reshape(-1, 1))
        try:
            yield
        finally:
            self._embedding = None

    def predict(self, y, f=None, q=None):
        if q is None or q.numel() != len(y):
            raise ValueError("One scalar quantile is required per image")
        with self.condition(q):
            conditioned = self.base.up2(self.base.features(y))
        return y + self.head(self.conditioner(conditioned, q))

    def load_original(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        expected = {
            k
            for k in self.state_dict()
            if k.startswith(("deep_embedding.", "deep_films."))
        }
        if set(missing) != expected or unexpected:
            raise ValueError(f"Original checkpoint mismatch: {missing}, {unexpected}")
