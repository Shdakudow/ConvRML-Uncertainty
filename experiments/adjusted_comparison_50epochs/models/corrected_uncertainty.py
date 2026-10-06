"""Corrected ConvRML variants; incompatible with the previous frozen/hybrid heads."""
import torch
from torch import nn
from models.shamus_rml import ShamusRML
from models.convnext_norm_film import FiLMBackbone
from models.formal_quantile import pinball

PROTOCOL = 'convrml_pinball_e2e_independent_shuffle_v1'

class PinballQUTCC(nn.Module):
    """A single quantile predictor, with median reconstruction and no MSE branch."""
    def __init__(self, base=None):
        super().__init__()
        self.base = FiLMBackbone(ShamusRML() if base is None else base)
        self.levels = (.05, .95)

    def forward(self, y, q=None):
        if q is None:
            q = y.new_full((len(y),), .5)
        return self.base(y, q)

    def bounds(self, y):
        return (self(y), self(y, y.new_full((len(y),), .05)),
                self(y, y.new_full((len(y),), .95)))

class EndToEndIm2Im(nn.Module):
    """RGB lower/center/upper outputs; the entire backbone remains trainable."""
    def __init__(self, base=None):
        super().__init__()
        self.base = ShamusRML() if base is None else base
        old = self.base.head
        self.channels = old.out_channels
        self.base.head = nn.Conv2d(old.in_channels, 3*self.channels,
                                  old.kernel_size, stride=old.stride,
                                  padding=old.padding, bias=old.bias is not None)
        self.base.requires_grad_(True)

    def forward(self, y):
        features = self.base.up2(self.base.features(y))
        residuals = self.base.head(features).reshape(len(y), 3, self.channels, *y.shape[-2:])
        return residuals + y[:, None]

    def bounds(self, y):
        outputs = self(y)
        return outputs[:, 1], outputs[:, 0], outputs[:, 2]


def training_loss(model, arm, y, x, q=None):
    if arm == 'conditioned':
        if q is None:
            raise ValueError('Supply the sampled per-image quantiles')
        return pinball(model(y, q).float(), x, q)
    if arm == 'im2im':
        prediction, lower, upper = model.bounds(y)
        return ((prediction.float()-x).square().mean()
                + pinball(lower.float(), x, x.new_full((len(x),), .05))
                + pinball(upper.float(), x, x.new_full((len(x),), .95)))
    if arm.startswith('mc_'):
        return (model(y).float()-x).square().mean()
    if arm.startswith('ensemble_'):
        mean, variance = model(y)
        mean, variance = mean.float(), variance.float()
        return .5*(variance.log()+(mean-x).square()/variance).mean()
    raise ValueError(arm)


def experiment_seeds(arm):
    if arm.startswith('ensemble_'):
        index = int(arm.split('_')[1])
        if index not in range(10):
            raise ValueError('Ensemble index must be 0..9')
    elif arm == 'conditioned':
        index = 100
    elif arm == 'im2im':
        index = 101
    elif arm in ('mc_005', 'mc_010', 'mc_020'):
        index = 102  # Matched initialization and order across dropout candidates.
    else:
        raise ValueError(arm)
    return dict(initialization=20261006+index, training=20271006+index,
                data=20281006+index, quantile=20291006+index)


def check_resume(checkpoint, arm, digest):
    if checkpoint.get('protocol') != PROTOCOL:
        raise ValueError('Cannot resume previous hybrid/frozen/shared-shuffle runs')
    if checkpoint.get('arm') != arm or checkpoint.get('manifest_sha256') != digest:
        raise ValueError('Resume method or data split mismatch')
    if checkpoint.get('seeds') != experiment_seeds(arm):
        raise ValueError('Resume seed policy mismatch')

class FullDropoutRML(ShamusRML):
    """Elementwise dropout in all 36 encoder blocks and 12 decoder conv stages."""
    def __init__(self, p=.1):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError('Expected 0 < dropout probability < 1')
        self.encoder_dropout = nn.ModuleDict()
        self.dropout_locations = []
        self._dropout_handles = []
        for name, block in list(self.encoder.named_modules()):
            if block.__class__.__name__ == 'ConvNeXtBlock':
                key = name.replace('.', '__')
                self.encoder_dropout[key] = nn.Dropout(p)
                self._dropout_handles.append(block.mlp.register_forward_hook(self._dropout_hook(key)))
                self.dropout_locations.append('encoder.'+name+'.mlp (before residual addition)')
        for name, block in [(f'refine.{i}', b) for i, b in enumerate(self.refine)] + [('up1.1', self.up1[1]), ('up2.1', self.up2[1])]:
            layers = []
            for layer in block.layers:
                layers.append(layer)
                if isinstance(layer, nn.GELU):
                    layers.append(nn.Dropout(p))
                    self.dropout_locations.append(name+'.after_activation_'+str(len(layers)-1))
            block.layers = nn.Sequential(*layers)
        if len(self.encoder_dropout) != 36 or len(self.dropout_locations) != 48:
            raise ValueError('Expected 36 encoder and 12 decoder dropout sites')

    def _dropout_hook(self, key):
        def hook(module, inputs, output):
            return self.encoder_dropout[key](output)
        return hook
