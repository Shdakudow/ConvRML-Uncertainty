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


def phase_one_settings():
    result = dict(
        training.SETTINGS, epochs=50, protocol="original50", schedule="original50"
    )
    for key in ("base_epochs", "continuation_initial_lr", "continuation_final_lr"):
        result.pop(key, None)
    return result


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
        total = 100 * 4500
        self.assertAlmostEqual(training.learning_rate(0, total), 1e-5)
        self.assertAlmostEqual(training.learning_rate(9 * 4500, total), 1e-4)
        self.assertAlmostEqual(training.learning_rate(total, total), 1e-6)
        self.assertLess(
            training.learning_rate(1000, total), training.learning_rate(1500, total)
        )
        self.assertGreater(
            training.learning_rate(60 * 4500, total),
            training.learning_rate(90 * 4500, total),
        )

    def test_50_epoch_extension_keeps_all_existing_warmup_learning_rates(self):
        with patch.dict(training.SETTINGS, phase_one_settings(), clear=True):
            for step in (0, 8000, 9000, 20000, 40500):
                extended_lr = training.learning_rate(step, 50 * 4500)
                with patch.dict(training.SETTINGS, warmup_fraction=0.3):
                    original_lr = training.learning_rate(step, 30 * 4500)
                self.assertAlmostEqual(extended_lr, original_lr, places=15)
            self.assertAlmostEqual(training.learning_rate(50 * 4500, 50 * 4500), 1e-9)

    def test_100_epoch_schedule_preserves_phase_one_and_uses_lower_rate_after_50(self):
        for step in (0, 10000, 40500, 100000, 224999):
            actual = training.learning_rate(step, 100 * 4500)
            with patch.dict(training.SETTINGS, phase_one_settings(), clear=True):
                original = training.learning_rate(step, 50 * 4500)
            self.assertAlmostEqual(actual, original, places=15)
        self.assertAlmostEqual(training.learning_rate(50 * 4500, 100 * 4500), 1e-5)
        self.assertAlmostEqual(training.learning_rate(75 * 4500, 100 * 4500), 5.5e-6)
        self.assertAlmostEqual(training.learning_rate(100 * 4500, 100 * 4500), 1e-6)

    def test_continuation_rejects_incomplete_epoch_50(self):
        state = dict(
            epoch=50,
            next_epoch=51,
            offset=0,
            global_step=225000,
            history=[dict(epoch=i) for i in range(1, 51)],
        )
        complete = dict(passed=True, epochs=50)
        training.validate_after50_checkpoint(state, complete)
        for changed in [
            dict(state, epoch=49),
            dict(state, next_epoch=50),
            dict(state, offset=4),
            dict(state, global_step=224999),
            dict(state, history=state["history"][:-1]),
        ]:
            with self.assertRaises(ValueError):
                training.validate_after50_checkpoint(changed, complete)
        with self.assertRaises(ValueError):
            training.validate_after50_checkpoint(state, dict(passed=False, epochs=50))

    def test_100_epoch_extension_rejects_higher_peak_and_changed_data(self):
        previous = dict(phase_one_settings(), pretrained=False)
        current = dict(training.SETTINGS, pretrained=False)
        training.validate_epoch_extension(previous, current)
        for changed in [
            dict(current, continuation_initial_lr=1e-4),
            dict(current, continuation_final_lr=0),
            dict(current, seed=0),
            dict(current, warmup_fraction=0.3),
        ]:
            with self.assertRaises(ValueError):
                training.validate_epoch_extension(previous, changed)

    def test_epoch_extension_rejects_other_training_identity_changes(self):
        current = dict(phase_one_settings(), pretrained=False)
        previous = dict(
            current,
            epochs=30,
            warmup_fraction=0.3,
            protocol="original30",
            schedule="original",
        )
        training.validate_epoch_extension(previous, current)
        for changed in [
            dict(current, seed=123),
            dict(current, manifest_sha256="different"),
            dict(current, effective_batch=8),
            dict(current, warmup_fraction=0.3),
        ]:
            with self.assertRaises(ValueError):
                training.validate_epoch_extension(previous, changed)

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
