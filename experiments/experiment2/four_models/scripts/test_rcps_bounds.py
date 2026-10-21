import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from scripts.rcps_bounds import select_factors, risk_threshold, hb_pvalue


class Tests(unittest.TestCase):
    def test_threshold_certifies_and_increases_with_sample_size(self):
        a = risk_threshold(2048)
        self.assertLess(hb_pvalue(a, 2048, 0.1), 0.05)
        self.assertGreaterEqual(hb_pvalue(a + 1e-6, 2048, 0.1), 0.05)
        self.assertLess(risk_threshold(200), a)
        self.assertLess(a, 0.1)

    def test_pixel_duplication_does_not_change_sample_count_or_certificate(self):
        x = np.random.default_rng(13).uniform(0, 4, (2048, 32))
        a, audit, losses = select_factors(x)
        b, other, _ = select_factors(np.repeat(x, 4, axis=1))
        self.assertEqual(a, b)
        self.assertEqual(audit, other)
        self.assertGreaterEqual(a["rcps_hb"], a["empirical"])
        self.assertTrue(np.all(losses["rcps_hb"] <= losses["empirical"]))

    def test_ties_and_insufficient_samples(self):
        factors, audit, _ = select_factors(np.ones((2048, 5)))
        self.assertEqual(factors["rcps_hb"], 1.0)
        self.assertEqual(audit["calibration_risks"]["rcps_hb"], 0.0)
        with self.assertRaises(ValueError):
            risk_threshold(2)


if __name__ == "__main__":
    unittest.main()
