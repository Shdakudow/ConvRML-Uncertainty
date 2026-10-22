"""Calibration matching, ties, clipping boundaries, and q-search failure cases."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.calibration.coverage90 import fit_tail_scale, search_quantiles


class TightCoverage(unittest.TestCase):
    def test_order_statistic_uses_smallest_feasible_scale(self):
        values = np.arange(1, 101, dtype=np.float32)
        result = fit_tail_scale(values, alpha=0.05)
        self.assertAlmostEqual(result["scale"], 95, places=3)
        self.assertEqual(result["risk"], 0.05)
        self.assertGreater(np.mean(values > np.nextafter(95.0, -np.inf)), 0.05)
        np.testing.assert_array_equal(values, np.arange(1, 101))

    def test_ties_do_not_fake_exact_coverage(self):
        result = fit_tail_scale(np.ones(100), alpha=0.05)
        self.assertAlmostEqual(result["scale"], 1)
        self.assertEqual(result["risk"], 0)

    def test_disk_backed_partition_fits_without_full_array_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = np.lib.format.open_memmap(
                Path(directory) / "scores.npy",
                mode="w+",
                dtype=np.float32,
                shape=(100, 100),
            )
            scores[:] = np.arange(10000).reshape(100, 100)[::-1]
            result = fit_tail_scale(scores, overwrite=True)
            json.dumps(result, allow_nan=False)
            self.assertAlmostEqual(result["risk"], 0.05)
            self.assertAlmostEqual(result["scale"], 9499, places=1)

    def test_clipped_scores_match_actual_bounds(self):
        rng = np.random.default_rng(18)
        target = rng.choice([0.0, 0.1, 0.5, 0.9, 1.0], 10000)
        pred = rng.uniform(-0.5, 1.5, len(target))
        dl, du = rng.uniform(0.01, 0.5, (2, len(target)))
        lower_scores, upper_scores = (pred - target) / dl, (target - pred) / du
        lower_scores[target == 1] = -np.inf
        upper_scores[target == 0] = -np.inf
        lower = fit_tail_scale(lower_scores)
        upper = fit_tail_scale(upper_scores)
        lo = np.clip(pred - lower["scale"] * dl, 0, 1)
        hi = np.clip(pred + upper["scale"] * du, 0, 1)
        actual = np.mean((target >= lo) & (target <= hi))
        self.assertAlmostEqual(actual, 1 - lower["risk"] - upper["risk"])
        self.assertTrue(0.9 <= actual <= 0.901)

    def test_nonnegative_scale_and_invalid_scores(self):
        self.assertEqual(fit_tail_scale(np.full(100, -np.inf))["scale"], 0)
        for invalid in ([], [np.nan], [np.inf]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fit_tail_scale(invalid)

    def test_q_search_refines_distinct_endpoints_without_scales(self):
        result = search_quantiles(
            lambda lo, hi: (lo**0.5, (1 - hi) ** 0.5),
            initial=(0.001, 0.999),
            coverage_tolerance=0.001,
        )
        coverage = 1 - result["lower_risk"] - result["upper_risk"]
        self.assertTrue(0.9 <= coverage <= 0.901)
        self.assertTrue(result["within_tolerance"])
        self.assertNotIn("scales", result)

    def test_infeasible_q_search_fails(self):
        with self.assertRaisesRegex(ValueError, "No feasible"):
            search_quantiles(lambda lo, hi: (0.1 + 0.2 * lo, 0.1 + 0.2 * (1 - hi)))

    def test_nonmonotone_q_response_fails(self):
        with self.assertRaisesRegex(ValueError, "nonmonotone"):
            search_quantiles(
                lambda lo, hi: (0.1 - 0.05 * lo, 0.1 * (1 - hi)), initial=(0.5, 0.5)
            )

    def test_plateau_is_reported_outside_tolerance(self):
        result = search_quantiles(
            lambda lo, hi: (0.0 if lo < 0.5 else 0.5, 0.0 if hi > 0.5 else 0.5)
        )
        self.assertFalse(result["within_tolerance"])
        self.assertEqual(result["coverage_lower_bound"], 1.0)


if __name__ == "__main__":
    unittest.main()
