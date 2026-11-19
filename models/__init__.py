"""The four models use the same ConvNeXt reconstruction backbone."""

from ConvNeXt import ConvRecon
from .ensemble import DeepEnsemble, DeepEnsembleMember
from .im2im import Im2im
from .mcdropout import MCDropout
from .qutcc import QUTCC


def make_model(name, height, width, dataset_size):
    def backbone(channels=3):
        return ConvRecon(3, height, width, out_channels=channels)

    factories = {
        "qutcc": lambda: QUTCC(backbone()),
        "im2im": lambda: Im2im(backbone(9)),  # Lower, center, and upper RGB images.
        "ensemble": lambda: DeepEnsemble(
            [DeepEnsembleMember(backbone(6)) for _ in range(10)]  # RGB mean and variance.
        ),
        "mcdropout": lambda: MCDropout(backbone(), dataset_size),
    }
    return factories[name]()
