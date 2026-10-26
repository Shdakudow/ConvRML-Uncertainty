"""Protect paper-only calibration, original numerics, and MC audit replay."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import infer as evaluate
from convrml import calibration
from archive.publication_support.calibration_legacy import (
    reference_calibration as original,
)


class PaperCalibration(unittest.TestCase):
    def test_rcps_scales_match_original_certified_branch_without_empirical_output(self):
        rng = np.random.default_rng(9)
        lower = rng.uniform(-1, 2, (2048, 16)).astype(np.float32)
        upper = rng.uniform(-1, 3, (2048, 16)).astype(np.float32)
        old_scales, old_audit, _ = original.select_asymmetric_factors(lower, upper)
        fitted = calibration.rcps_scales(lower, upper)
        self.assertEqual(fitted["scales"], old_scales["rcps_hb"])
        for side in ("lower", "upper"):
            self.assertEqual(
                fitted["certificate"]["tail_hb_pvalues"][side],
                old_audit["tail_hb_pvalues"][side],
            )
        self.assertNotIn("empirical", fitted)
        self.assertEqual(set(fitted["scales"]), {"lower", "upper"})

    def test_algorithm1_keeps_original_selected_quantiles_and_finite_n_target(self):
        risk = lambda lo, hi: (lo**0.5, (1 - hi) ** 0.5)
        old = original.paper_quantiles(risk, 3000)
        new = calibration.paper_quantiles(risk, 3000)
        for key in ("lower_q", "upper_q", "per_tail_target", "trace"):
            self.assertEqual(new[key], old[key])
        self.assertNotIn("scales", new)

    def test_small_calibration_set_and_infeasible_q_fail_explicitly(self):
        with self.assertRaisesRegex(calibration.CalibrationFailure, "Insufficient"):
            calibration.rcps_scales(np.ones((4, 16)), np.ones((4, 16)))
        with self.assertRaisesRegex(calibration.CalibrationFailure, "No feasible"):
            calibration.paper_quantiles(lambda lo, hi: (0.1, 0.1), 3000)

    def test_raw_im2im_uses_rcps_and_mc_audit_replays_the_fit_randomness(self):
        torch.manual_seed(7)
        pairs = torch.rand(128, 2, 3, 8, 8)
        draws, states = [], []

        def predict(y):
            spread = 0.1 + torch.rand_like(y) * 0.02
            draws.append(spread.clone())
            states.append(torch.get_rng_state())
            center = torch.full_like(y, 0.5)
            return center, center - spread, center + spread

        with tempfile.TemporaryDirectory() as output, torch.inference_mode():
            result = evaluate.fit_calibration(
                pairs,
                predict,
                None,
                dict(method="im2im", experiment=3, microbatch=32),
                torch.device("cpu"),
                0,
                Path(output),
            )
        self.assertEqual(result["procedure"], "asymmetric RCPS-HB")
        self.assertTrue(result["has_hb_certificate"])
        self.assertIsNone(result["endpoint_clipping"])
        self.assertEqual(len(draws), 8)
        for fit, audit in zip(draws[:4], draws[4:]):
            self.assertTrue(torch.equal(fit, audit))
        self.assertTrue(torch.equal(torch.get_rng_state(), states[3]))

    def test_qutcc_uses_quantile_calibration_with_no_endpoint_scales(self):
        class Quantiles:
            def __call__(self, y, q):
                return torch.zeros_like(y) + q[:, None, None, None]

            def bounds(self, y, lo, hi):
                return tuple(self(y, y.new_full((len(y),), q)) for q in (0.5, lo, hi))

        torch.manual_seed(11)
        pairs = torch.rand(128, 2, 3, 8, 8)
        model = Quantiles()
        with tempfile.TemporaryDirectory() as output, torch.inference_mode():
            result = evaluate.fit_calibration(
                pairs,
                None,
                model,
                dict(method="qutcc", microbatch=32),
                torch.device("cpu"),
                0,
                Path(output),
            )
        self.assertIn("Algorithm 1", result["procedure"])
        self.assertNotIn("scales", result)
        self.assertFalse(result["has_hb_certificate"])
        self.assertLess(result["lower_q"], result["upper_q"])
        self.assertNotEqual((result["lower_q"], result["upper_q"]), (0.05, 0.95))
        self.assertGreaterEqual(result["calibration_coverage"], 0.9)


if __name__ == "__main__":
    unittest.main()
