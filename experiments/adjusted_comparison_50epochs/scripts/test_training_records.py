import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.training_records import split_bounds, save_milestone, check_resume_manifest


class RecordsTest(unittest.TestCase):
    def test_nonstandard_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps({"train": [0] * 7, "val": [0] * 3, "test": [0] * 2})
            )
            self.assertEqual(split_bounds(root, "test"), (10, 2, 12))
            self.assertEqual(split_bounds(root, "val"), (7, 3, 12))

    def test_weights_preserved_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 1)
            weights = model.weight.detach().clone()
            for epoch in range(1, 7):
                save_milestone(directory, model, epoch, 5, manifest_sha256="abc")
            files = list(Path(directory).rglob("*.pt"))
            self.assertEqual([p.name for p in files], ["epoch_005.pt"])
            with torch.no_grad():
                model.weight.add_(1)
            state = torch.load(files[0], weights_only=False)
            torch.testing.assert_close(state["model"]["weight"], weights)
            self.assertEqual(state["manifest_sha256"], "abc")
            with self.assertRaises(FileExistsError):
                save_milestone(directory, model, 5, 5)

    def test_resume_provenance(self):
        check_resume_manifest({"manifest_sha256": "a"}, "a")
        check_resume_manifest({}, "a", "a")
        for state in ({}, {"manifest_sha256": "b"}):
            with self.assertRaises(ValueError):
                check_resume_manifest(state, "a")


if __name__ == "__main__":
    unittest.main()
