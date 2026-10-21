import unittest
import numpy as np
from metrics import correlation, maps


class Metrics(unittest.TestCase):
    def test_monotonic_nonlinear(self):
        x = np.arange(20, dtype=float)
        r = correlation(x, x**3)
        self.assertAlmostEqual(r["spearman"], 1)
        self.assertLess(r["pearson"], 0.95)

    def test_constant_is_undefined(self):
        self.assertIsNone(correlation(np.zeros(9), np.arange(9))["pearson"])

    def test_absolute_before_channel_average(self):
        p = np.array([[[1.0]], [[0.0]], [[0.5]]])
        t = np.array([[[0.0]], [[1.0]], [[0.5]]])
        u, e = maps(p, t, np.zeros_like(p), np.ones_like(p))
        self.assertAlmostEqual(float(e[0, 0]), 2 / 3, places=6)
        self.assertEqual(float(u[0, 0]), 1.0)

    def test_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            maps(
                np.zeros((3, 4, 4)),
                np.zeros((3, 3, 4)),
                np.zeros((3, 4, 4)),
                np.zeros((3, 4, 4)),
            )


if __name__ == "__main__":
    unittest.main()
