"""Explicit full-state continuation; epoch-50 source checkpoints stay immutable."""

import hashlib, math
from pathlib import Path
import torch
from data import PROJECT, identity

PARENT = PROJECT / "experiments/qutcc_paper_style_20261018/results/raw"


def path(method):
    return PARENT / method / "latest.pt"


def parent_identity(method):
    p = path(method)
    with p.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return dict(path=str(p), sha256=digest, epoch=50)


def learning_rate(global_step, steps_per_epoch):
    fraction = (global_step - 50 * steps_per_epoch) / (30 * steps_per_epoch)
    assert 0 <= fraction <= 1, (global_step, steps_per_epoch)
    return 1e-9 + (0.00001 - 1e-9) * 0.5 * (1 + math.cos(math.pi * fraction))


def load_initial(model, optimizer, rng, method, return_checkpoint=False):
    ck = torch.load(path(method), map_location="cpu", weights_only=False)
    config = ck["config"]
    assert config["protocol"] == "paper_qutcc_convnext_uniform_pinball_20261018_v1"
    assert config["method"] == method and config["camera"] == "raw"
    assert config["dataset_identity"] == identity("raw")
    assert (
        ck["next_epoch"] == 51
        and ck["offset"] == 0
        and ck["history"][-1]["epoch"] == 50
    )
    assert ck["global_step"] == 50 * 4500
    if return_checkpoint:
        return ck
    model.load_state_dict(ck["model"], strict=True)
    optimizer.load_state_dict(ck["optimizer"])
    rng.set_state(ck["q_rng"])
    torch.set_rng_state(ck["cpu_rng"])
    torch.cuda.set_rng_state(ck["cuda_rng"])
    assert optimizer.state, "Optimizer moments were not retained"
    return None
