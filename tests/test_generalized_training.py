"""Check deterministic validation conditions, split separation, and target safety."""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch.utils.data import Dataset


SOURCE = Path(__file__).resolve().parents[1] / "experiments/generalized_inputs/train.py"
SPEC = importlib.util.spec_from_file_location("generalized_training", SOURCE)
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


class ToyPairs(Dataset):
    def __init__(self):
        self.pairs = torch.rand(
            (12, 2, 3, 64, 64), generator=torch.Generator().manual_seed(1)
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


class GeneralizedTraining(unittest.TestCase):
    def test_validation_panel_is_fixed_balanced_and_preserves_original_targets(self):
        torch.set_num_threads(1)
        source = ToyPairs()
        original = source.pairs.clone()
        first = training.ValidationPanel(source, 3)
        replay = training.ValidationPanel(source, 3)
        self.assertEqual(first.indices, replay.indices)
        self.assertEqual(len(first), 30)
        for index in range(len(first)):
            measurement, target, condition = first[index]
            replay_measurement, replay_target, replay_condition = replay[index]
            torch.testing.assert_close(measurement, replay_measurement, rtol=0, atol=0)
            torch.testing.assert_close(target, replay_target, rtol=0, atol=0)
            self.assertEqual(condition, replay_condition)
            self.assertEqual(condition, index // 3)
            source_index = first.indices[index % 3]
            torch.testing.assert_close(
                target, original[source_index, 1], rtol=0, atol=0
            )
            if condition == 0:
                torch.testing.assert_close(
                    measurement, original[source_index, 0], rtol=0, atol=0
                )
        torch.testing.assert_close(source.pairs, original, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
