"""Check the publication API, paired-file conventions and native intervals."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convrml.data import ImagePairs, NumpyPairs
from experiments.experiment2.models.ensemble import DeepEnsemble
from experiments.experiment2.models.mc_dropout import MCDropout


class PairedData(unittest.TestCase):
    def test_numpy_split_order_and_separate_calibration_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = np.stack(
                [np.full((2, 3, 4, 5), value, dtype=np.uint8) for value in (10, 20, 30)]
            )
            np.save(root / "pairs.npy", pairs)
            manifest = {role: [dict(id=role)] for role in ("train", "val", "test")}
            (root / "manifest.json").write_text(json.dumps(manifest))
            for role, value in zip(manifest, pairs):
                dataset = NumpyPairs(root / "pairs.npy", root / "manifest.json", role)
                self.assertEqual(dataset.ids, (role,))
                torch.testing.assert_close(
                    dataset[0], torch.from_numpy(value).float() / 255, rtol=0, atol=0
                )
            with self.assertRaises(ValueError):
                NumpyPairs(root / "pairs.npy", root / "manifest.json", "calibration")
            np.save(root / "calibration.npy", pairs[:1])
            (root / "calibration.json").write_text(
                json.dumps({"calibration": [{"id": "heldout"}]})
            )
            dataset = NumpyPairs(
                root / "calibration.npy", root / "calibration.json", "calibration"
            )
            self.assertEqual(dataset.ids, ("heldout",))

    def test_image_pairs_keep_rectangular_dimensions_rgb_values_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            y = np.full((7, 11, 3), 80, dtype=np.uint8)
            x = np.full((7, 11, 3), 160, dtype=np.uint8)
            Image.fromarray(y).save(root / "measurement.png")
            Image.fromarray(x).save(root / "target.png")
            row = dict(
                id="scene",
                measurement=dict(path="measurement.png"),
                target=dict(path=str(root / "target.png")),
            )
            (root / "split.json").write_text(json.dumps({"roles": {"test": [row]}}))
            dataset = ImagePairs(root / "split.json", "test")
            actual = dataset[0]
            self.assertEqual(actual.shape, (2, 3, 7, 11))
            self.assertEqual(dataset.ids, ("scene",))
            expected = torch.stack(
                [
                    torch.from_numpy(value).permute(2, 0, 1).float() / 255
                    for value in (y, x)
                ]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_experiment_folders_have_only_the_six_public_models(self):
        for experiment, methods in [
            ("experiment2", {"im2im", "ensemble", "mc_dropout", "qutcc"}),
            ("experiment3", {"im2im", "qutcc"}),
        ]:
            root = ROOT / "experiments" / experiment
            self.assertEqual(
                {p.name for p in root.iterdir() if p.name != "__pycache__"},
                {"README.md", "models"},
            )
            self.assertEqual(
                {
                    p.stem
                    for p in (root / "models").glob("*.py")
                    if p.stem != "__init__"
                },
                methods,
            )


class NativeIntervals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compare with the actual saved-run predictor, rather than a copied formula.
        path = (
            ROOT
            / "archive/reproduction/experiments/experiment2/scripts/export_corrected_milestone.py"
        )
        spec = importlib.util.spec_from_file_location("saved_run_predictor", path)
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)

    def test_mc_30_pass_bounds_match_saved_run_predictor_exactly(self):
        class FakeNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropouts = nn.ModuleList([nn.Dropout(0.05) for _ in range(48)])

            def cuda(self):
                return self

            def forward(self, y):
                return self.dropouts[0](y + torch.rand_like(y))

        metadata = dict(epoch=50, seeds={}, history=[])
        network = FakeNetwork().eval()
        with patch.object(
            self.reference, "load_model", return_value=(network, metadata)
        ):
            predict, _ = self.reference.predictor("rml", "mc_005", 50)
        y = torch.ones(1, 3, 3, 4)
        torch.manual_seed(42)
        expected = predict(y)
        torch.manual_seed(42)
        actual = MCDropout.bounds(network, y)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertFalse(network.training)
        self.assertTrue(all(not layer.training for layer in network.dropouts))

    def test_five_member_bounds_match_saved_run_predictor_exactly(self):
        class FakeMember(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.index = index

            def cuda(self):
                return self

            def forward(self, y):
                return y + self.index / 10, torch.full_like(y, 0.01 * (self.index + 1))

        members = [FakeMember(i) for i in range(5)]
        metadata = dict(epoch=50, seeds={}, history=[])
        with patch.object(
            self.reference, "load_model", side_effect=[(m, metadata) for m in members]
        ):
            predict, _ = self.reference.predictor("rml", "ensemble5", 50)
        y = torch.rand(1, 3, 3, 4)
        expected = predict(y)
        actual = DeepEnsemble(members).bounds(y)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
