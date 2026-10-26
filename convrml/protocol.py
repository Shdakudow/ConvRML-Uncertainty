"""Recorded experiment settings, model construction and dataset identity checks."""

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import torch

from .data import ImagePairs, NumpyPairs

SETTINGS_PATH = Path(__file__).with_name("experiment_settings.json")


def settings():
    return json.loads(SETTINGS_PATH.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    models = [
        path
        for experiment in ("parallel_lensless", "raw_diffusercam")
        for path in (root / "experiments" / experiment / "models").glob("*.py")
    ]
    files = sorted(list(Path(__file__).parent.glob("*.py")) + models + [SETTINGS_PATH])
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_shape(dataset, dataset_name, *, smoke=False):
    expected = tuple(settings()["datasets"][dataset_name]["shape"])
    if not smoke and tuple(dataset[0][0].shape) != expected:
        raise ValueError("Image dimensions differ from the recorded experiment")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def profile(experiment, dataset, method, member=None):
    config = settings()
    data = config["datasets"][dataset]
    if data["experiment"] != experiment:
        raise ValueError("Dataset does not belong to the requested experiment")
    if method not in config["methods"][f"experiment{experiment}"]:
        raise ValueError("Method is not part of this experiment")
    if (method == "ensemble") != (member is not None):
        raise ValueError(
            "Ensemble training/loading requires a member index; other models do not"
        )
    if member is not None and member not in range(10):
        raise ValueError("Ensemble member index must be 0..9")
    family = (
        "adaptation"
        if dataset == "raw_corrupted"
        else ("paper" if method == "qutcc" or experiment == 3 else "baseline")
    )
    result = dict(config["profiles"][family])
    if data["loader"] == "images":
        result["microbatch"] = 1
    policy = config["seed_policy"]
    if family == "baseline":
        index = member if method == "ensemble" else policy["arm_index"][method]
        seeds = {
            name: policy["baseline_" + name] + index
            for name in ("initialization", "training", "data", "quantile")
        }
        shuffle = "persistent_generator"
    else:
        seed = policy["paper_base"] + data["camera_index"]
        seeds = dict(
            initialization=seed, training=seed, data=seed, quantile=seed + 1000
        )
        shuffle = "per_epoch_permutation"
    return dict(
        result,
        schema=config["schema"],
        experiment=experiment,
        dataset=dataset,
        method=method,
        member=member,
        seeds=seeds,
        shuffle=shuffle,
        dropout=config["dropout_probability"],
        family=family,
    )


def make_model(run):
    """Keep original parameter names and distinct ensemble initialization seeds."""
    torch.manual_seed(run["seeds"]["initialization"])
    if run["experiment"] == 2:
        from experiments.parallel_lensless.models import (
            Im2im,
            QUTCC,
            MCDropout,
            EnsembleMember,
        )
    else:
        from experiments.raw_diffusercam.models import Im2im, QUTCC
    if run["method"] == "qutcc":
        return QUTCC()
    if run["method"] == "im2im":
        return Im2im()
    if run["method"] == "mc_dropout":
        return MCDropout(run["dropout"])
    return EnsembleMember()


def dataset_for(dataset, manifest, pairs, split):
    if settings()["datasets"][dataset]["loader"] == "numpy":
        if pairs is None:
            raise ValueError(
                "Parallel lensless experiment requires --pairs for its NumPy cache"
            )
        return NumpyPairs(pairs, manifest, split)
    if pairs is not None:
        raise ValueError(
            "Raw DiffuserCam experiment reads paired image paths, not a NumPy cache"
        )
    return ImagePairs(manifest, split)


def id_digest(ids):
    return hashlib.sha256(
        json.dumps(list(map(str, ids)), separators=(",", ":")).encode()
    ).hexdigest()


def audit_manifest(manifest, dataset, *, smoke=False):
    """Validate IDs before loading pixels; count/hash checks may relax only in smoke mode."""
    value = json.loads(Path(manifest).read_text())
    roles = value.get("roles", value)
    expected = settings()["datasets"][dataset]
    identities, seen = {}, set()
    for role in ("train", "val", "test", "calibration"):
        if role not in roles:
            continue
        ids = [str(row["id"]) for row in roles[role]]
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Empty, duplicated or overlapping split IDs")
        seen.update(ids)
        digest = id_digest(ids)
        if not smoke and len(ids) != expected["counts"][role]:
            raise ValueError(
                "Split count differs from the recorded experiment: " + role
            )
        known = expected.get("id_hashes", {}).get(role)
        if not smoke and known and digest != known:
            raise ValueError(
                "Split IDs/order differ from the recorded experiment: " + role
            )
        identities[role] = dict(count=len(ids), ids_sha256=digest)
    if not identities:
        raise ValueError("Manifest has no recognized splits")
    return dict(manifest_sha256=sha256(manifest), splits=identities)


def autocast(run, device):
    return (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if run["precision"] == "bfloat16"
        else nullcontext()
    )


def learning_rate(step, total_steps, run):
    peak = run["peak_lr"]
    if run["schedule"] == "constant":
        return peak
    fraction, low = step / total_steps, peak / 10
    return (
        low + fraction / 0.3 * (peak - low)
        if fraction <= 0.3
        else (
            low / 1e4
            + (peak - low / 1e4) / 2 * (math.cos(math.pi * (fraction - 0.3) / 0.7) + 1)
        )
    )


def center(model, method, y):
    if method == "ensemble":
        return model(y)[0]
    if method == "mc_dropout":
        return model(y)
    return model.center(y) if hasattr(model, "center") else model(y)[:, 1]


def environment():
    from importlib.metadata import version
    import platform
    from . import __version__

    return dict(
        package=__version__,
        python=platform.python_version(),
        torch=torch.__version__,
        torchvision=version("torchvision"),
        timm=version("timm"),
        numpy=version("numpy"),
        Pillow=version("Pillow"),
        torchmetrics=version("torchmetrics"),
        scipy=version("scipy"),
        cuda=torch.version.cuda,
    )


def hardware(device):
    import platform

    return dict(
        device=str(device),
        accelerator=(
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.machine()
        ),
    )
