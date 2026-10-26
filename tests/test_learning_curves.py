"""Protect raw/clipped conventions and checkpoint identity in curve exports."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "archive/publication_support/tools/plot_learning_curves.py"
)
SPEC = importlib.util.spec_from_file_location("curves", SOURCE)
curves = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curves)


class LearningCurveReports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, report):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))

    def test_raw_and_clipped_are_distinct(self):
        raw = dict(ssim=0.7, psnr=25, coverage=0.9, width=0.2)
        clipped = dict(
            ssim=0.71,
            psnr=26,
            coverage=0.92,
            width=0.18,
            lpips=0.12,
            pearson=None,
            spearman=0.6,
            mae=0.05,
        )
        self.write(
            "epoch_005/report.json",
            dict(
                epoch=5,
                test_count=2000,
                metrics=dict(raw=raw, clipped=clipped),
                calibration=dict(lower_q=0.03, upper_q=0.97),
            ),
        )
        row = curves.read_series("QUTCC", self.root)[0]
        self.assertEqual(row["width"], 0.18)
        self.assertEqual(row["test_count"], 2000)
        self.assertEqual(row["lower_q"], 0.03)
        self.assertEqual(row["lpips"], 0.12)
        self.assertEqual(row["pearson"], "")
        self.assertEqual(row["spearman"], 0.6)
        self.assertEqual(
            curves.read_series("QUTCC", self.root, "raw")[0]["coverage"], 0.9
        )

    def test_clipped_baseline_cannot_be_reported_as_raw(self):
        self.write(
            "epoch_010.json",
            dict(
                epoch=10,
                clipped=True,
                test_count=5000,
                metrics=dict(ssim=0.8, psnr=30, coverage=0.91, width=0.1),
                factors=dict(lower=1.1, upper=1.3),
            ),
        )
        row = curves.read_series("MC", self.root)[0]
        self.assertEqual(row["upper_scale"], 1.3)
        with self.assertRaisesRegex(ValueError, "convention"):
            curves.read_series("MC", self.root, "raw")

    def test_duplicate_epoch_rejected(self):
        report = dict(
            epoch=5,
            clipped=True,
            metrics=dict(ssim=0.8, psnr=30, coverage=0.9, width=0.1),
        )
        self.write("epoch_005.json", report)
        self.write("other/epoch_005/report.json", report)
        with self.assertRaisesRegex(ValueError, "duplicate epoch"):
            curves.read_series("mixed", self.root)


if __name__ == "__main__":
    unittest.main()
