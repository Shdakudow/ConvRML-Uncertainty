"""Dataset provenance and compact immutable evaluation checkpoints."""

import hashlib
import json
from pathlib import Path

import torch


def manifest_digest(data):
    return hashlib.sha256((Path(data) / "manifest.json").read_bytes()).hexdigest()


def split_bounds(data, role):
    manifest = json.loads((Path(data) / "manifest.json").read_text())
    counts = {name: len(manifest[name]) for name in ("train", "val", "test")}
    if any(n <= 0 for n in counts.values()):
        raise ValueError("All dataset splits must be nonempty")
    start = {
        "train": 0,
        "val": counts["train"],
        "test": counts["train"] + counts["val"],
    }[role]
    return start, counts[role], sum(counts.values())


def save_milestone(out, model, epoch, interval, **metadata):
    """Save weights/provenance for inference; latest.pt retains optimizer/RNG."""
    if interval <= 0 or epoch % interval:
        return
    dest = Path(out) / "checkpoints" / f"epoch_{epoch:03d}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite milestone: {dest}")
    temporary = dest.with_suffix(".tmp")
    torch.save({**metadata, "model": model.state_dict(), "epoch": epoch}, temporary)
    temporary.replace(dest)


def check_resume_manifest(checkpoint, digest, legacy_digest=None):
    previous = checkpoint.get("manifest_sha256", legacy_digest)
    if previous != digest:
        raise ValueError("Resume dataset identity mismatch or missing provenance")
