"""QUTCC: condition the ConvNeXt reconstruction on one quantile q per image."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import pinball_loss


class FiLM(nn.Module):
    """Learn a quantile-dependent scale and offset after normalization."""

    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Sequential(nn.SiLU(), nn.Linear(512, 2 * channels))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, x, embedding, channels_last=False):
        scale, offset = self.projection(embedding).chunk(2, dim=1)
        shape = (len(x), 1, 1, -1) if channels_last else (len(x), -1, 1, 1)
        return x * (1 + scale.reshape(shape)) + offset.reshape(shape)


class QUTCC(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(128, 512), nn.SiLU(), nn.Linear(512, 512)
        )
        self.encoder_film = nn.ModuleList(
            [
                nn.ModuleList([FiLM(block.dwconv.out_channels) for block in stage])
                for stage in backbone.encoder.stages
            ]
        )
        self.decoder_film = nn.ModuleList(
            [
                FiLM(layer.num_features)
                for layer in backbone.decoder.FNN1007.f
                if isinstance(layer, nn.BatchNorm2d)
            ]
        )

    def forward(self, image, q=0.5):
        """Return an RGB quantile image; q=0.5 is the reconstruction median."""
        q = torch.as_tensor(q, device=image.device, dtype=image.dtype).expand(
            len(image)
        )
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(64, device=image.device) / 64
        )
        angles = q[:, None] * frequencies[None]
        embedding = self.embedding(torch.cat((angles.cos(), angles.sin()), dim=1))
        height, width = image.shape[-2:]
        x = F.pad(image, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")

        features = []
        for index, stage in enumerate(self.backbone.encoder.stages):
            x = self.backbone.encoder.downsample_layers[index](x)
            for block, film in zip(stage, self.encoder_film[index]):
                x = block(x, film=film, embedding=embedding)
            features.append(getattr(self.backbone.encoder, f"norm{index}")(x))

        size = (
            self.backbone.decoder.original_height,
            self.backbone.decoder.original_width,
        )
        x = torch.cat(
            [
                F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
                for feature in features
            ],
            dim=1,
        )
        film_index = 0
        for layer in self.backbone.decoder.FNN1007.f:
            x = layer(x)
            if isinstance(layer, nn.BatchNorm2d):
                x = self.decoder_film[film_index](x, embedding)
                film_index += 1
        return x

    def center(self, image):
        return self(image, 0.5)

    def bounds(self, image, lower_q=0.05, upper_q=0.95):
        return self.center(image), self(image, lower_q), self(image, upper_q)

    def loss(self, image, target):
        q = torch.rand(len(image), device=image.device)
        return pinball_loss(self(image, q), target, q)
