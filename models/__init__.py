"""Four shared uncertainty methods; datasets select loaders and configurations."""

import torch

from .im2im import Im2im
from .ensemble import EnsembleMember, DeepEnsemble
from .mc_dropout import MCDropout
from .qutcc import QUTCC

__all__ = ["Im2im", "EnsembleMember", "DeepEnsemble", "MCDropout", "QUTCC"]


def make_model(run):
    """Construct dataset-independent model classes with recorded initialization seeds."""
    torch.manual_seed(run["seeds"]["initialization"])

    if run["method"] == "qutcc":
        return QUTCC()
    if run["method"] == "im2im":
        return Im2im()
    if run["method"] == "mc_dropout":
        return MCDropout(run["dropout"])
    if run["method"] == "ensemble":
        return EnsembleMember()
    raise ValueError("Unknown model method: " + run["method"])


def center(model, method, y):
    """One-pass validation prediction, shape (N,3,H,W).

    With model.eval(), MC dropout is deterministic here; final inference instead
    averages stochastic passes. An ensemble training run validates its member,
    not an aggregate. Im2im returns its center head; QUTCC queries q=0.5.
    """
    if method == "ensemble":
        return model(y)[0]
    if method == "mc_dropout":
        return model(y)
    return model.center(y)
