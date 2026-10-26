"""Compare the release-native interval implementation to saved-run prediction."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.experiment2.models.ensemble import DeepEnsemble
from experiments.experiment2.models.mc_dropout import MCDropout


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
