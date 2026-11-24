"""Shared-lambda grid calibration adapted from Shdakudow/code-for-quant.

Reference: commit 0e164119fc81e1d0c9cb1a889d2a2f0f2525f081,
evaluation/calibration.py and evaluation/bounds.py. The local adapter caches
one RGB batch per file so the search reuses identical stochastic predictions
without requiring the full calibration split to fit in GPU or host memory.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import xlogy
from scipy.stats import binom

from models.common import gaussian_interval_from_mean_variance


def HB_mu_plus(muhat: float, n: int, delta: float) -> float:
    def _tailprob(mu):
        clipped_mu = np.clip(mu, 1e-12, 1 - 1e-12)
        divergence = xlogy(muhat, muhat / clipped_mu) + xlogy(
            1 - muhat,
            (1 - muhat) / (1 - clipped_mu),
        )
        hoeffding_mu = -n * divergence if muhat < mu else 0.0
        bentkus_mu = np.log(max(binom.cdf(np.floor(n * muhat), n, mu), 1e-10)) + 1
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)

    if _tailprob(1 - 1e-10) > 0:
        return 1.0
    return float(brentq(_tailprob, max(muhat, 1e-12), 1 - 1e-10, maxiter=1000))


def fraction_missed_loss(pset, label):
    misses = (pset[0] > label) | (pset[2] < label)
    return misses.flatten(start_dim=1).float().mean(dim=1)


def interval_sets_from_output(
    output: torch.Tensor,
    lam: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = output.clone()
    output[:, 0, :, :] = torch.minimum(output[:, 0, :, :], output[:, 1, :, :] - 1e-6)
    output[:, 2, :, :] = torch.maximum(output[:, 2, :, :], output[:, 1, :, :] + 1e-6)
    prediction = output[:, 1, :, :]
    lower_edge = prediction - lam * (prediction - output[:, 0, :, :])
    upper_edge = lam * (output[:, 2, :, :] - prediction) + prediction
    upper_edge = torch.maximum(upper_edge, prediction + 1e-6)
    lower_edge = torch.minimum(lower_edge, prediction - 1e-6)
    return lower_edge, prediction, upper_edge


def scaled_bounds(center, lower, upper, scale):
    output = torch.stack((lower, center, upper), dim=1)
    lower, _, upper = interval_sets_from_output(output, scale)
    return lower, upper


def select_shared_lambda_from_outputs(
    batch_files, n_images, device, dtype,
    alpha: float = 0.1,
    delta: float = 0.1,
    min_lam: float = 0,
    max_lam: float = 6,
    num_lam: int = 1000,
) -> tuple[float, float, float]:
    """Select the first lambda below the paper's adjusted empirical-risk target."""
    risk_target = alpha - (1 - alpha) / n_images
    lambdas = torch.linspace(
        min_lam,
        max_lam,
        num_lam,
        device=device,
        dtype=dtype,
    )

    def evaluate(index: int) -> tuple[float, float, float]:
        lambda_value = lambdas[index]
        losses = []
        for path in batch_files:
            output, label = torch.load(path, mmap=True, weights_only=True)
            bounds = interval_sets_from_output(output.to(device), lambda_value)
            losses.append(fraction_missed_loss(bounds, label.to(device)))
        risk = float(torch.cat(losses).mean().item())
        ucb = float(HB_mu_plus(risk, n_images, delta))
        print(f"Lambda: {lambda_value.item():.6f} | Rhat: {risk:.6f} | RhatPlus: {ucb:.6f}", flush=True)
        return float(lambda_value.item()), risk, ucb

    low, high = 0, num_lam - 1
    while low < high:
        midpoint = (low + high) // 2
        candidate = evaluate(midpoint)
        if candidate[1] < risk_target:
            high = midpoint
        else:
            low = midpoint + 1

    selected_lambda, selected_risk, selected_ucb = evaluate(low)
    if selected_risk >= risk_target:
        raise RuntimeError(f"No lambda in [{min_lam}, {max_lam}] reached risk target {risk_target:.6f}")
    print(f"Optimal lambda found: {selected_lambda:.6f}", flush=True)
    return selected_lambda, selected_risk, selected_ucb


class GlobalGridCalibration:
    def __init__(self, model, method):
        self.model = model
        self.method = method
        self.max_scale, self.grid_size = {"mcdropout": (50, 2000), "ensemble": (10, 1000)}[method]
        self.parameters = {}

    def _native_bounds(self, image):
        if self.method == "ensemble":
            return self.model.bounds(image)
        draws = self.model.predict_samples(image)
        variance, center = torch.var_mean(draws, dim=0, unbiased=False)
        lower, upper = gaussian_interval_from_mean_variance(center, variance)
        return center, lower, upper

    @torch.inference_mode()
    def fit(self, loader, device):
        """Cache native intervals once, then fit one multiplier on the calibration split."""
        self.model.eval()
        n_images = 0
        with TemporaryDirectory() as directory:
            batch_files = []
            for image, target in loader:
                center, lower, upper = self._native_bounds(image.to(device))
                output = torch.stack((lower, center, upper), dim=1)
                path = Path(directory) / f"{len(batch_files)}.pt"
                torch.save((output.cpu(), target.cpu()), path)
                batch_files.append(path)
                n_images += len(target)
            if not n_images:
                raise ValueError("The calibration split is empty.")
            scale, risk, ucb = select_shared_lambda_from_outputs(
                batch_files, n_images, device, output.dtype,
                max_lam=self.max_scale, num_lam=self.grid_size,
            )
        self.parameters = dict(
            procedure="Empirical global grid", scale=scale,
            grid_size=self.grid_size, max_scale=self.max_scale, n_images=n_images,
            alpha=0.1, target_coverage=0.9, adjusted_risk_target=0.1 - 0.9 / n_images,
            calibration_coverage=1 - risk, hb_upper_bound=ucb, delta=0.1,
        )
        return self.parameters

    @torch.inference_mode()
    def predict(self, image):
        center, lower, upper = self._native_bounds(image)
        lower, upper = scaled_bounds(center, lower, upper, self.parameters["scale"])
        return center, lower, upper
