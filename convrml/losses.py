"""Pixelwise quantile loss shared by Im2im and QUTCC."""

import torch


def pinball(prediction, target, q):
    """Average pinball loss; q contains one scalar per image, shared by its pixels."""
    error = target - prediction
    q = q.reshape(-1, 1, 1, 1)
    return torch.maximum(q * error, (q - 1) * error).mean()
