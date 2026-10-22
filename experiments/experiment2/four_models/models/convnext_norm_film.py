"""QUTCC-style scale/shift after normalization in ConvNeXt and its FPN.

The backbone is still Shamus ConvNeXtV2-Base. Hooks modulate the internal
residual branch (not the completed residual block). Sequential forwards only.
"""

from contextlib import contextmanager
import copy
import torch
from torch import nn
from models.shamus_rml import ShamusRML
from baselines.formal_rml import FormalRML
from models.qutcc_original.nn import timestep_embedding


from models._shared_film import NormFiLM, FiLMBackbone


class ConvNeXtFiLMQuantile(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.base = FiLMBackbone(ShamusRML())
        self.levels = (0.05, 0.95)
        if kind == "fixed":
            self.base.requires_grad_(False).eval()
            self.head = nn.Conv2d(256, 6, 3, padding=1)
            with torch.no_grad():
                self.head.weight.copy_(self.base.head.weight.repeat(2, 1, 1, 1))
                self.head.bias.copy_(self.base.head.bias.repeat(2))
        elif kind == "conditioned":
            self.head = copy.deepcopy(self.base.head)
        else:
            raise ValueError(kind)

    def train(self, mode=True):
        super().train(mode)
        if self.kind == "fixed":
            self.base.eval()
        return self

    def features(self, y):
        f = self.base.features(y)
        return f, y + self.base.head(f)

    def predict(self, y, f=None, q=None):
        if self.kind == "fixed":
            return self.head(f).reshape(len(y), 2, 3, *y.shape[-2:]) + y[:, None]
        return y + self.head(self.base.features(y, q))

    def bounds(self, y):
        if self.kind == "fixed":
            f, r = self.features(y)
            p = self.predict(y, f)
            return r, p[:, 0], p[:, 1]
        return (
            self.base(y),
            self.predict(y, q=y.new_full((len(y),), 0.05)),
            self.predict(y, q=y.new_full((len(y),), 0.95)),
        )
