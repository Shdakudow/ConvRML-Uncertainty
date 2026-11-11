"""Check correlation conventions, rank ties, undefined maps and LPIPS integration."""

from unittest.mock import patch
import unittest

import numpy as np
import torch
from scipy import stats

from evaluation.metrics import (
    PerceptualMetric,
    finite_mean,
    interval_metrics,
    uncertainty_error_correlations,
)


class Metrics(unittest.TestCase):
    def test_correlations_match_scipy_per_image_with_rank_ties(self):
        width = torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 8.0]).reshape(
            2, 1, 2, 2
        )
        error = torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0]).reshape_as(width)
        result = uncertainty_error_correlations(width, error)
        for i in range(2):
            u, e = (
                width[i].double().numpy().flatten(),
                error[i].double().numpy().flatten(),
            )
            self.assertAlmostEqual(
                float(result["pearson"][i]), stats.pearsonr(u, e).statistic
            )
            self.assertAlmostEqual(
                float(result["spearman"][i]), stats.spearmanr(u, e).statistic
            )

    def test_constant_maps_are_undefined_and_aggregates_count_valid_images(self):
        result = uncertainty_error_correlations(
            torch.ones(1, 3, 4, 4), torch.rand(1, 3, 4, 4)
        )
        for value in result.values():
            self.assertTrue(torch.isnan(value).all())
        self.assertEqual(finite_mean([None, np.nan, 0.5, -0.5]), (0.0, 2))
        self.assertEqual(finite_mean([None, np.nan]), (None, 0))

    def test_raw_and_clipped_correlations_use_their_own_error_and_width(self):
        torch.manual_seed(4)
        target = torch.rand(1, 3, 16, 16)
        prediction = torch.rand_like(target) * 2 - 0.5
        lo, hi = prediction - 0.2, prediction + 0.4
        for clipped in (False, True):
            result = interval_metrics(prediction, lo, hi, target, clipped=clipped)
            p, l, u = [
                value.clamp(0, 1) if clipped else value
                for value in (prediction, lo, hi)
            ]
            expected = uncertainty_error_correlations(u - l, (p - target).abs())
            for key in expected:
                torch.testing.assert_close(result[key], expected[key], equal_nan=True)

    def test_lpips_uses_clipped_inputs_resets_state_and_preserves_rng(self):
        class Stub(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                torch.rand(3)
                self.options = kwargs
                self.resets = 0

            def forward(self, first, second):
                self.first, self.second = first.clone(), second.clone()
                return (first - second).abs().mean((1, 2, 3))

            def reset(self):
                self.resets += 1

        torch.manual_seed(9)
        state = torch.get_rng_state()
        with patch(
            "torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity", Stub
        ):
            metric = PerceptualMetric(torch.device("cpu"))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        prediction = torch.full((2, 3, 64, 64), 1.5)
        target = torch.full_like(prediction, 0.25)
        torch.testing.assert_close(metric(prediction, target), torch.full((2,), 0.75))
        self.assertEqual(
            metric.metric.options,
            dict(net_type="alex", normalize=True, reduction="none"),
        )
        self.assertEqual(metric.metric.resets, 1)
        self.assertTrue(torch.equal(metric.metric.first, prediction.clamp(0, 1)))
        self.assertTrue(torch.equal(metric.metric.second, target))


if __name__ == "__main__":
    unittest.main()
