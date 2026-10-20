"""Full ConvNeXtV2-FPN baselines, trained independently on real RGB pairs."""

import torch
from torch import nn
from models.shamus_rml import ShamusRML


class FormalRML(ShamusRML):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method not in ("ensemble", "mc"):
            raise ValueError(method)
        if method == "ensemble":
            self.variance_head = nn.Conv2d(256, 3, 3, padding=1)
            nn.init.zeros_(self.variance_head.weight)
            nn.init.constant_(self.variance_head.bias, -4.6)
        else:
            # Executed dropout in the decoder, present during training and MC inference.
            for block in list(self.refine) + [self.up1[1], self.up2[1]]:
                layers = []
                for layer in block.layers:
                    layers.append(layer)
                    if isinstance(layer, nn.GELU):
                        layers.append(nn.Dropout2d(0.1))
                block.layers = nn.Sequential(*layers)

    def forward(self, x):
        f = self.up2(self.features(x))
        mean = x + self.head(f)
        if self.method == "ensemble":
            return torch.cat((mean, self.variance_head(f)), 1)
        return mean
