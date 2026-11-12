"""Calibrate frozen models on held-out images.

KRCPS fits baseline spatial interval offsets. QUTCCCalibration selects quantile inputs.
Both target 90% coverage; KRCPS uses the Hoeffding-Bentkus certificate.
"""

import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import xlogy
from scipy.stats import binom
import torch
from torch.utils.data import DataLoader, Subset

from Calibration.krcps_optimization import SETTINGS, _optimize_group_lambdas


class KRCPS:
    """Fit spatial additive offsets with K-RCPS, then certify held-out risk.

    Group optimization uses the existing code-for-quant settings. The final
    HB step targets alpha=0.1 with delta=0.1 (90% confidence). Optimization
    images are excluded from this final check.
    """

    def __init__(self, model):
        self.model = model
        self.parameters = {}

    @staticmethod
    def _hb_pvalue(risk, n):
        alpha = SETTINGS["alpha"]
        if risk >= alpha:
            return 1.0
        divergence = xlogy(risk, risk / alpha) + xlogy(
            1 - risk, (1 - risk) / (1 - alpha)
        )
        return min(
            1.0,
            math.exp(-n * divergence),
            math.e * binom.cdf(math.ceil(n * risk), n, alpha),
        )

    def _accepted_risk(self, n):
        lower, upper = 0.0, SETTINGS["alpha"]
        for _ in range(SETTINGS["hb_iterations"]):
            middle = (lower + upper) / 2
            if self._hb_pvalue(middle, n) < SETTINGS["delta"]:
                lower = middle
            else:
                upper = middle
        return lower

    @staticmethod
    def _risk(target, lower, upper, expansion):
        lower = (lower - expansion).clamp(0, 1)
        upper = (upper + expansion).clamp(0, 1)
        return float(((target < lower) | (target > upper)).float().mean())

    @torch.inference_mode()
    def fit(self, loader, device):
        self.model.eval()
        device = torch.device(device)
        n = len(loader.dataset)
        n_opt = SETTINGS["n_opt"]
        order = torch.randperm(n).tolist()
        ordered_loader = DataLoader(
            Subset(loader.dataset, order),
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
        )
        shape = (n, *loader.dataset[0][1].shape)

        # Cache predictions on disk once. CPU tensors can view this storage;
        # CUDA calibration transfers the cached endpoints after inference ends.
        with tempfile.TemporaryDirectory() as folder:
            arrays = [
                np.lib.format.open_memmap(
                    Path(folder) / f"{name}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                )
                for name in ("target", "lower", "upper")
            ]
            offset = 0
            for measurement, target in ordered_loader:
                _, lower, upper = self.model.bounds(measurement.to(device))
                values = (
                    target,
                    torch.minimum(lower, upper),
                    torch.maximum(lower, upper),
                )
                for array, value in zip(arrays, values):
                    array[offset : offset + len(target)] = value.cpu().numpy()
                offset += len(target)

            rng = np.random.default_rng(torch.initial_seed())
            membership, group_offsets = _optimize_group_lambdas(
                *(torch.from_numpy(array[:n_opt]) for array in arrays),
                epsilon=SETTINGS["alpha"],
                gamma=SETTINGS["gamma"],
                k=SETTINGS["k"],
                prob_size=SETTINGS["prob_size"],
                device=device,
                rng=rng,
            )
            expansion = membership.to(device) @ group_offsets + SETTINGS["lambda_max"]
            target, lower, upper = (
                torch.from_numpy(array[n_opt:]).to(device) for array in arrays
            )
            n_cal = n - n_opt
            threshold = self._accepted_risk(n_cal)
            risk = self._risk(target, lower, upper, expansion)
            selected = expansion.clone()
            selected_risk = risk
            while risk <= threshold:
                selected = expansion.clone()
                selected_risk = risk
                if torch.all(expansion == 0):
                    break
                expansion = (expansion - SETTINGS["stepsize"]).clamp_min(0)
                risk = self._risk(target, lower, upper, expansion)

            pvalue = self._hb_pvalue(selected_risk, n_cal)
            self.parameters = dict(
                procedure="K-RCPS with held-out Hoeffding-Bentkus calibration",
                lambda_map=selected.cpu(),
                alpha=SETTINGS["alpha"],
                delta=SETTINGS["delta"],
                target_coverage=1 - SETTINGS["alpha"],
                confidence=1 - SETTINGS["delta"],
                n_images=n_cal,
                n_optimization_images=n_opt,
                optimization_indices=order[:n_opt],
                calibration_indices=order[n_opt:],
                calibration_coverage=1 - selected_risk,
                crossing=0.0,
                target_met=selected_risk <= SETTINGS["alpha"],
                hb_pvalue=float(pvalue),
                certificate_passed=bool(pvalue < SETTINGS["delta"]),
                hb_iterations=SETTINGS["hb_iterations"],
                endpoint_clipping=[0, 1],
            )
            for array in arrays:
                array._mmap.close()
        return self.parameters

    def predict(self, image):
        center, first, second = self.model.bounds(image)
        expansion = self.parameters["lambda_map"].to(image.device)
        lower = (torch.minimum(first, second) - expansion).clamp(0, 1)
        upper = (torch.maximum(first, second) + expansion).clamp(0, 1)
        return center, lower, upper


class QUTCCCalibration:
    """Select two q inputs using QUTCC's finite-image corrected tail budget."""

    def __init__(self, model):
        self.model = model
        self.parameters = {}

    @torch.inference_mode()
    def fit(self, loader, device):
        self.model.eval()
        n = len(loader.dataset)
        target_risk = (0.1 - 0.9 / n) / 2
        left, right = [0.0, 0.0], [1.0, 1.0]
        selected = [0.0, 1.0]
        for _ in range(12):
            qs = [(lo + hi) / 2 for lo, hi in zip(left, right)]
            misses, pixels = [0, 0], 0
            for measurement, target in loader:
                measurement, target = measurement.to(device), target.to(device)
                lower = self.model(measurement, qs[0])
                upper = self.model(measurement, qs[1])
                misses[0] += int((target < lower).sum())
                misses[1] += int((target > upper).sum())
                pixels += target.numel()
            if misses[0] / pixels <= target_risk:
                left[0] = selected[0] = qs[0]
            else:
                right[0] = qs[0]
            if misses[1] / pixels <= target_risk:
                right[1] = selected[1] = qs[1]
            else:
                left[1] = qs[1]
        self.parameters = dict(
            lower_q=selected[0],
            upper_q=selected[1],
            procedure="QUTCC quantile search",
        )

        # Measure the actual joint coverage of the selected quantile bounds.
        covered = total = crossed = 0
        for measurement, target in loader:
            measurement, target = measurement.to(device), target.to(device)
            _, lower, upper = self.predict(measurement)
            covered += int(((target >= lower) & (target <= upper)).sum())
            crossed += int((lower > upper).sum())
            total += target.numel()
        self.parameters.update(
            n_images=n,
            target_coverage=0.9,
            calibration_coverage=covered / total,
            crossing=crossed / total,
            target_met=covered / total >= 0.9,
        )
        return self.parameters

    def predict(self, image):
        return self.model.bounds(
            image, self.parameters["lower_q"], self.parameters["upper_q"]
        )
