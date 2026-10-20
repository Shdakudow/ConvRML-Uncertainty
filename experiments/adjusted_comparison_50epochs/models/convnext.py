# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# Adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv because of the groups=dim
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
    ):
        super().__init__()

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(
                    self, f"norm{i}"
                )  # convenient way to avoid using if else's to do norm1, norm2, norm3, etc
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# from Lensless Imaging Transformer
# https://github.com/BobPXX/Lensless_Imaging_Transformer/tree/main


class FNN(nn.Module):
    def __init__(
        self,
        sum_encoder_embed_dims=1440,
        rec_channels=3,
        decoder_channels=(512, 256, 128, 64),
    ):
        """Convolutional reconstruction head.

        The default widths are the published ConvRML head.  Keeping them as an
        argument makes small, explicit capacity experiments possible while
        retaining checkpoint compatibility for the default model.
        """
        super(FNN, self).__init__()
        self.rec_channels = rec_channels
        if len(decoder_channels) != 4 or any(
            channel <= 0 for channel in decoder_channels
        ):
            raise ValueError(
                "decoder_channels must contain four positive channel widths"
            )

        self.dropout = nn.Dropout(0.3)
        self.act_Lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.act_relu = nn.ReLU()
        self.act_tanh = nn.Tanh()

        layers = []
        in_channels = sum_encoder_embed_dims
        for out_channels in decoder_channels:
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels
        layers.append(
            nn.Conv2d(
                in_channels, self.rec_channels, kernel_size=3, padding=1, bias=False
            )
        )
        self.f = nn.Sequential(*layers)

    def forward(self, x):

        x = self.f(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        sum_encoder_embed_dims=1440,
        rec_channels=3,
        original_height=300,
        original_width=480,
        decoder_channels=(512, 256, 128, 64),
    ):
        super(Decoder, self).__init__()

        self.original_height = original_height
        self.original_width = original_width
        self.FNN1007 = FNN(
            sum_encoder_embed_dims=sum_encoder_embed_dims,
            rec_channels=rec_channels,
            decoder_channels=decoder_channels,
        )

    def fuse_features(self, x):
        """Upsample and concatenate encoder features before reconstruction.

        This is intentionally parameter-free.  Keeping it separate lets a
        conditional branch modulate the same fused feature map while the
        published reconstruction head (``FNN1007``) remains unchanged.
        """
        s1, s2, s3, s4 = x

        s1 = F.interpolate(
            s1,
            size=(self.original_height, self.original_width),
            mode="bilinear",
            align_corners=False,
        )  # doing something similar to U-Net (using residuals from previous layers, but they are all of different sizes, so need to be interpolated)
        s2 = F.interpolate(
            s2,
            size=(self.original_height, self.original_width),
            mode="bilinear",
            align_corners=False,
        )
        s3 = F.interpolate(
            s3,
            size=(self.original_height, self.original_width),
            mode="bilinear",
            align_corners=False,
        )
        s4 = F.interpolate(
            s4,
            size=(self.original_height, self.original_width),
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat((s1, s2, s3, s4), 1).type(s1.dtype)

    def forward(self, x):
        return self.FNN1007(self.fuse_features(x))


class ConvRecon(nn.Module):
    def __init__(
        self,
        n_channels,
        height,
        width,
        model_size="tiny",
        pretrained=False,
        in_22k=False,
        drop_path_rate=0.2,
        layer_scale_init_value=1.0,
        decoder_channels=(512, 256, 128, 64),
    ):
        super().__init__()
        if model_size == "tiny":
            self.encoder = convnext_tiny(
                pretrained,
                in_22k,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                in_chans=n_channels,
            )  # in_22k is boolean for if model is trained on imagenet 22k. If false, pretrained on imagenet 1k
        elif model_size == "small":
            self.encoder = convnext_small(
                pretrained,
                in_22k,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                in_chans=n_channels,
            )
        else:
            raise ValueError("model_size must be 'tiny' or 'small'")
        self.decoder = Decoder(
            rec_channels=n_channels,
            original_height=height,
            original_width=width,
            decoder_channels=decoder_channels,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class QuantileConditioner(nn.Module):
    """Condition an image feature map on a scalar quantile level.

    The condition combines three complementary mechanisms:
    (1) an MLP embedding of ``q``; (2) FiLM scale and shift; and (3) a gated
    residual produced from a spatially broadcast condition map.  The FiLM and
    residual paths start as identities, so copying the reconstruction head to
    the quantile head gives a stable, meaningful initialization.
    """

    def __init__(self, feature_channels, embedding_dim=128, condition_channels=32):
        super().__init__()
        self.feature_channels = feature_channels
        self.quantile_mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
        )
        self.film = nn.Linear(embedding_dim, 2 * feature_channels)
        self.condition_map = nn.Linear(embedding_dim, condition_channels)
        self.residual = nn.Conv2d(
            feature_channels + condition_channels, feature_channels, kernel_size=1
        )
        self.gate = nn.Linear(embedding_dim, feature_channels)

        # Start at an identity map. The quantile head can therefore begin as a
        # copy of the MSE reconstruction head and specialize during training.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, features, quantile):
        if quantile.ndim == 1:
            quantile = quantile[:, None]
        if quantile.ndim != 2 or quantile.shape[1] != 1:
            raise ValueError(
                f"quantile must have shape (batch,) or (batch, 1), got {tuple(quantile.shape)}"
            )
        if quantile.shape[0] != features.shape[0]:
            raise ValueError("quantile batch size must match feature batch size")

        embedding = self.quantile_mlp(quantile.to(dtype=features.dtype))
        scale, shift = self.film(embedding).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        film_features = features * (1.0 + scale) + shift

        condition = self.condition_map(embedding)[:, :, None, None]
        condition = condition.expand(-1, -1, features.shape[-2], features.shape[-1])
        residual = self.residual(torch.cat((film_features, condition), dim=1))
        gate = torch.sigmoid(self.gate(embedding))[:, :, None, None]
        return film_features + gate * residual


class QuantileConvRecon(nn.Module):
    """ConvNeXt reconstruction model with an unchanged MSE and a quantile head.

    ``reconstruction`` is exactly the normal ConvRecon decoder output. The
    second output, ``quantile_reconstruction``, shares encoder features but is
    conditioned on q before an identically shaped convolutional head.
    """

    def __init__(
        self,
        n_channels,
        height,
        width,
        model_size="tiny",
        pretrained=False,
        in_22k=False,
        drop_path_rate=0.2,
        layer_scale_init_value=1.0,
        decoder_channels=(512, 256, 128, 64),
        quantile_embedding_dim=128,
        quantile_condition_channels=32,
        quantile_normalization="batch",
    ):
        super().__init__()
        if model_size == "tiny":
            self.encoder = convnext_tiny(
                pretrained,
                in_22k,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                in_chans=n_channels,
            )
            fused_channels = 96 + 192 + 384 + 768
        elif model_size == "small":
            self.encoder = convnext_small(
                pretrained,
                in_22k,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                in_chans=n_channels,
            )
            fused_channels = 96 + 192 + 384 + 768
        else:
            raise ValueError("model_size must be 'tiny' or 'small'")

        # Same normal reconstruction head and state-dict path as ConvRecon.
        self.decoder = Decoder(
            rec_channels=n_channels,
            original_height=height,
            original_width=width,
            decoder_channels=decoder_channels,
        )
        self.quantile_conditioner = QuantileConditioner(
            fused_channels,
            embedding_dim=quantile_embedding_dim,
            condition_channels=quantile_condition_channels,
        )
        self.quantile_head = FNN(
            sum_encoder_embed_dims=fused_channels,
            rec_channels=n_channels,
            decoder_channels=decoder_channels,
        )
        self.quantile_head.load_state_dict(self.decoder.FNN1007.state_dict())
        if quantile_normalization == "group":
            for index, layer in enumerate(self.quantile_head.f):
                if isinstance(layer, nn.BatchNorm2d):
                    self.quantile_head.f[index] = nn.GroupNorm(8, layer.num_features)
        elif quantile_normalization != "batch":
            raise ValueError("quantile_normalization must be batch or group")

    def _encode_fused(self, x):
        features = self.encoder(x)
        return self.decoder.fuse_features(features)

    def forward_quantiles(self, x, quantiles):
        """Predict several quantiles while evaluating the shared encoder once."""
        fused_features = self._encode_fused(x)
        reconstruction = self.decoder.FNN1007(fused_features)
        predictions = [
            self.quantile_head(self.quantile_conditioner(fused_features, quantile))
            for quantile in quantiles
        ]
        return reconstruction, predictions

    def forward(self, x, quantile):
        reconstruction, predictions = self.forward_quantiles(x, [quantile])
        return reconstruction, predictions[0]


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = (
            model_urls["convnext_tiny_22k"]
            if in_22k
            else model_urls["convnext_tiny_1k"]
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = (
            model_urls["convnext_small_22k"]
            if in_22k
            else model_urls["convnext_small_1k"]
        )
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = (
            model_urls["convnext_base_22k"]
            if in_22k
            else model_urls["convnext_base_1k"]
        )
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = (
            model_urls["convnext_large_22k"]
            if in_22k
            else model_urls["convnext_large_1k"]
        )
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert (
            in_22k
        ), "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls["convnext_xlarge_22k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
