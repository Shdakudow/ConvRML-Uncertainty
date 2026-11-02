"""Exp 4 fairness, calibration failure, and publication integrity checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
import numpy as np

from experiments.experiment4_demo import conditions as c
from experiments.experiment4_demo import evaluate as e
from experiments.experiment4_demo import publish as p


class Experiment4Tests(unittest.TestCase):
    def test_calibration_mixture_matches_training_family_weights(self):
        from collections import Counter

        counts = Counter(c.calibration_conditions())
        self.assertEqual(sum(counts.values()), 3000)
        self.assertEqual(len(counts), 11)
        for name, count in counts.items():
            self.assertEqual(count, 150 if name.startswith("exposure_") else 300)
        self.assertEqual(c.calibration_conditions(), c.calibration_conditions())

    def test_all_inputs_repeat_without_mutating_original(self):
        image = torch.rand(1, 3, 64, 128, generator=torch.Generator().manual_seed(1))
        before = image.clone()
        for condition in c.IDS:
            first = c.modify(image, condition, split="test", identifier="scene1")
            second = c.modify(image, condition, split="test", identifier="scene1")
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(image, before, rtol=0, atol=0)
            self.assertEqual(first.shape, image.shape)
            self.assertTrue(bool(torch.all((first >= 0) & (first <= 1))))
        other = c.modify(image, "gaussian", split="test", identifier="scene2")
        self.assertFalse(
            torch.equal(
                other, c.modify(image, "gaussian", split="test", identifier="scene1")
            )
        )

    def test_global_exposure_and_exact_dead_fraction(self):
        image = torch.full((1, 3, 100, 100), 0.4)
        for name, gain in [("exposure_050", 0.5), ("exposure_150", 1.5)]:
            changed = c.modify(image, name, split="test", identifier="one")
            torch.testing.assert_close(changed, image * gain)
        dead = c.modify(image, "dead_pixels", split="test", identifier="one")
        self.assertEqual(int((dead[0, 0] == 0).sum()), 1000)
        self.assertTrue(torch.equal(dead[0, 0] == 0, dead[0, 1] == 0))

    def test_paired_loader_keeps_target_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sensor = np.full((64, 64, 3), 100, dtype=np.uint8)
            target = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
            Image.fromarray(sensor).save(root / "sensor.png")
            Image.fromarray(target).save(root / "target.png")
            (root / "split.json").write_text(
                json.dumps(
                    dict(
                        roles=dict(
                            test=[
                                dict(
                                    id="one",
                                    measurement=dict(path="sensor.png"),
                                    target=dict(path="target.png"),
                                )
                            ]
                        )
                    )
                )
            )
            dataset = e.ModifiedPairs(root / "split.json", "test", "gaussian")
            changed, truth, original = dataset[0]
            torch.testing.assert_close(
                truth,
                torch.from_numpy(target).permute(2, 0, 1).float() / 255,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                original,
                torch.from_numpy(sensor).permute(2, 0, 1).float() / 255,
                rtol=0,
                atol=0,
            )
            self.assertFalse(torch.equal(changed, original))

    def test_unattainable_qutcc_is_a_visible_failure(self):
        class ToyQ:
            def __call__(self, y, q=None):
                return torch.full_like(y, 0.4 if float(q[0]) < 0.5 else 0.6)

            def bounds(self, y, lo, hi):
                return (
                    torch.full_like(y, 0.5),
                    self(y, y.new_tensor([lo])),
                    self(y, y.new_tensor([hi])),
                )

        y = torch.zeros(1, 3, 4, 4)
        batch = (y, torch.full_like(y, 0.9), y)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(e, "loader", return_value=[batch]),
            patch.object(e, "move", side_effect=lambda b: b[:2]),
        ):
            fit = e.calibrate(ToyQ(), "qutcc_clean50", [batch], Path(temp))
            self.assertEqual(fit["status"], "target_unattainable")
            self.assertFalse(fit["target_met"])
            self.assertEqual(fit["calibration_coverage"], 0)
            self.assertEqual((fit["lower_q"], fit["upper_q"]), (0, 1))

    def test_wrong_baseline_scores_stop_export(self):
        records = [
            dict(id=str(i), clipped=dict(psnr=20.0, ssim=0.8, mae=0.02))
            for i in range(50)
        ]
        rows = [dict(psnr=20.0, ssim=0.8, mae=0.02) for _ in range(50)]
        e.parity(rows, dict(qutcc=records), "qutcc", [str(i) for i in range(50)])
        rows[5] = dict(rows[5], ssim=0.7)
        with self.assertRaisesRegex(ValueError, "parity failed"):
            e.parity(rows, dict(qutcc=records), "qutcc", [str(i) for i in range(50)])

    def test_incomplete_artifacts_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "READY.json").write_text(
                json.dumps(dict(schema="experiment4-v1", passed=True, files={}))
            )
            with self.assertRaisesRegex(ValueError, "inventory"):
                p.verify(root)

    def test_navigation_is_idempotent_and_preserves_existing_links(self):
        source = Path(__file__).resolve().parents[1] / "experiments/experiment4_demo"
        original = "const routes={legacy:'./'};\n  const state=new URLSearchParams(location.search);\n"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "navigation.js"
            path.write_text(original)
            p.navigation(path)
            first = path.read_text()
            p.navigation(path)
            self.assertEqual(first, path.read_text())
            self.assertIn("legacy:'./'", first)
            self.assertEqual(first.count("exp4:'experiment4/'"), 1)


if __name__ == "__main__":
    unittest.main()
