"""Strict checkpoint loading with portable experiment identity and file hashes."""

import hashlib
from pathlib import Path

import torch

from .protocol import make_model, settings


def read_checkpoint(path):
    """Hash and deserialize the same open file, even during atomic replacement."""
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or "model" not in checkpoint
        or "epoch" not in checkpoint
    ):
        raise ValueError("Expected a checkpoint with model state and completed epoch")
    return checkpoint, digest.hexdigest()


def identity(checkpoint):
    meta = checkpoint.get("run") or checkpoint.get("config") or checkpoint
    arm = meta.get("arm", meta.get("method"))
    if isinstance(arm, str) and arm.startswith("ensemble_"):
        method, member = "ensemble", int(arm.split("_")[1])
    elif isinstance(arm, str) and arm.startswith("mc_"):
        if arm != "mc_005":
            raise ValueError("The published MC comparison uses p=0.05")
        method, member = "mc_dropout", None
    else:
        method, member = arm, meta.get("member")
    dataset = meta.get("dataset")
    if dataset is None:
        camera = meta.get("camera")
        if camera == "raw":
            dataset = (
                "raw_corrupted"
                if "corrupt" in meta.get("protocol", "")
                else "raw_clean"
            )
        else:
            dataset = camera
    return dict(
        method=method,
        dataset=dataset,
        member=member,
        epoch=checkpoint["epoch"],
        seeds=meta.get("seeds", {}),
        smoke=checkpoint.get("smoke", False),
        protocol=meta.get("protocol", meta.get("schema", checkpoint.get("schema"))),
    )


def validate_identity(checkpoint, run, *, smoke=False):
    """Reject superseded training protocols or mismatched benchmark provenance."""
    info = identity(checkpoint)
    for key in ("method", "dataset", "member"):
        if info[key] != run[key]:
            raise ValueError(f"Checkpoint {key} does not match this evaluation")
    protocols = settings()["checkpoint_protocols"][run["family"]] + [run["schema"]]
    if info["protocol"] not in protocols:
        raise ValueError("Checkpoint belongs to a different training protocol")
    if info["protocol"] != run["schema"]:
        meta = checkpoint.get("config") or checkpoint
        expected = settings()["datasets"][run["dataset"]]["source_manifest_hashes"]
        stored = set(meta.get("dataset_identity", {}).values())
        if run["family"] == "baseline":
            expected = expected[:1]
            stored.add(meta.get("manifest_sha256"))
        if not set(expected) <= stored:
            raise ValueError(
                "Checkpoint training data identity differs from the recorded benchmark"
            )
    if info["smoke"] and not smoke:
        raise ValueError("A smoke checkpoint cannot be presented as a full experiment")
    return info


def load_model(path, run, device, *, smoke=False, factory=None):
    checkpoint, digest = read_checkpoint(path)
    info = validate_identity(checkpoint, run, smoke=smoke)
    model = (factory or make_model)(run)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, dict(info, file=Path(path).name, sha256=digest)
