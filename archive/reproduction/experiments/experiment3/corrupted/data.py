"""Read the exact fixed, reviewed dataset. No new corruptions during training."""

import hashlib, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from base_data import identity as base_identity, audit as base_audit, PROJECT

PARENT = PROJECT / "experiments/qutcc_paper_style_20261018"
PARENT_EPOCH = 30
ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "dataset/manifest.json"


def identity(camera):
    assert camera == "raw"
    result = base_identity(camera)
    for p in [ROOT / "protocol.json", MANIFEST]:
        result[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def audit(camera):
    assert camera == "raw"
    expected = base_audit(camera)
    manifest = json.loads(MANIFEST.read_text())
    seen = set()
    assert manifest["counts"] == expected
    for role, count in expected.items():
        rows = manifest["roles"][role]
        assert len(rows) == count
        for row in rows:
            identifier = str(row["id"])
            assert identifier not in seen
            seen.add(identifier)
            assert Path(row["measurement"]["path"]).is_file()
            assert Path(row["target"]["path"]).is_file()
    return expected


def load_parent(model, method):
    path = (
        PARENT / "results/raw" / method / "checkpoints" / f"epoch_{PARENT_EPOCH:03d}.pt"
    )
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["epoch"] == PARENT_EPOCH
    assert (
        ck["config"]["protocol"] == "paper_qutcc_convnext_uniform_pinball_20261018_v1"
    )
    assert ck["config"]["dataset_identity"] == base_identity("raw")
    assert ck["config"]["method"] == method
    model.load_state_dict(ck["model"], strict=True)
    return dict(
        path=str(path),
        sha256=digest,
        epoch=PARENT_EPOCH,
        protocol=ck["config"]["protocol"],
    )


class Pairs(Dataset):
    def __init__(self, camera, role):
        assert camera == "raw"
        assert role in ("train", "val", "test", "calibration")
        self.rows = json.loads(MANIFEST.read_text())["roles"][role]
        self.camera = camera

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        pass  # Fixed reviewed images; only sample order changes.

    def __getitem__(self, index):
        out = []
        for key in ("measurement", "target"):
            with Image.open(self.rows[index][key]["path"]) as im:
                out.append(
                    torch.from_numpy(np.array(im.convert("RGB"), copy=True))
                    .permute(2, 0, 1)
                    .float()
                    / 255
                )
        assert out[0].shape == out[1].shape
        return torch.stack(out)
