"""Shared ConvNeXtV2-Base encoder and FPN residual decoder."""

import torch
from torch import nn
from torch.nn import functional as F
import timm


class Refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class ConvNeXtFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k", pretrained=False, features_only=True
        )
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, 256, 1) for c in [128, 256, 512, 1024]]
        )
        self.refine = nn.ModuleList([Refine() for _ in range(4)])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Refine()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Refine()
        )
        self.head = nn.Conv2d(256, 3, 3, padding=1)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def features(self, x):
        features = self.encoder((x - self.mean) / self.std)
        d = self.refine[3](self.lateral[3](features[3]))
        for i in [2, 1, 0]:
            d = self.refine[i](
                F.interpolate(
                    d, size=features[i].shape[-2:], mode="bilinear", align_corners=False
                )
                + self.lateral[i](features[i])
            )
        return self.up1(d)

    def reconstruct(self, x, features):
        return x + self.head(self.up2(features))

    def forward(self, x):
        return self.reconstruct(x, self.features(x))
