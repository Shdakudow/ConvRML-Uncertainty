"""Check deterministic validation conditions, split separation, and target safety."""

import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

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
    def test_epoch_zero_state_replays_exactly_and_rejects_trained_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fresh = torch.nn.Linear(4, 3)
            expected = training.prepare_initial_weights(fresh, output, "qutcc", True)
            recreated = torch.nn.Linear(4, 3)
            self.assertNotEqual(expected, training.weights_hash(recreated))
            actual = training.prepare_initial_weights(recreated, output, "qutcc", False)
            self.assertEqual(expected, actual)
            self.assertTrue(torch.equal(fresh.weight, recreated.weight))
            with self.assertRaises(FileExistsError):
                training.prepare_initial_weights(fresh, output, "qutcc", True)
            path = output / "initial_random.pt"
            state = torch.load(path, weights_only=False)
            state["trained_steps"] = 1
            torch.save(state, path)
            with self.assertRaises(ValueError):
                training.prepare_initial_weights(recreated, output, "qutcc", False)

    def test_scratch_initialization_repeats_seed_and_has_no_parent_weights(self):
        # Tiny constructors exercise the initialization path without allocating
        # multiple ConvNeXt models on the CPU; the GPU smoke checks the real model.
        for method, constructor in [("qutcc", "QUTCC"), ("im2im", "Im2im")]:
            with patch.object(
                training, constructor, side_effect=lambda: torch.nn.Linear(4, 3)
            ):
                first = training.initialize_model(method)
                torch.manual_seed(999)
                replay = training.initialize_model(method)
            self.assertEqual(
                training.weights_hash(first), training.weights_hash(replay)
            )
            self.assertTrue(torch.equal(first.weight, replay.weight))
            self.assertEqual(
                training.SETTINGS["initialization"],
                "fresh_random_no_pretrained_weights",
            )
            self.assertNotIn("parent_epoch", training.SETTINGS)
            self.assertNotIn("parent_sha256", training.SETTINGS)

    def test_scratch_schedule_warms_up_and_decays_without_segment_resets(self):
        total = 10000
        self.assertAlmostEqual(training.learning_rate(0, total), 1e-5)
        self.assertAlmostEqual(training.learning_rate(3000, total), 1e-4)
        self.assertAlmostEqual(training.learning_rate(total, total), 1e-9)
        self.assertLess(
            training.learning_rate(1000, total), training.learning_rate(2000, total)
        )
        self.assertGreater(
            training.learning_rate(5000, total), training.learning_rate(8000, total)
        )

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
