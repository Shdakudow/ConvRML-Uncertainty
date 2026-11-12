"""Group-offset optimization
Copyright (c) 2023 Sulam-Group. Distributed under the MIT license in
KRCPS_LICENSE.txt.
"""

from __future__ import annotations

import logging
import cvxpy as cp
import numpy as np
import torch
from tqdm import tqdm

SETTINGS = {
    "alpha": 0.1,
    "delta": 0.1,
    "hb_iterations": 40,
    "lambda_max": 0.5,
    "stepsize": 0.002,
    "n_opt": 10,
    "gamma": [
        0.25,
        0.26724138,
        0.28448276,
        0.30172414,
        0.31896552,
        0.3362069,
        0.35344828,
        0.37068966,
        0.38793103,
        0.40517241,
        0.42241379,
        0.43965517,
        0.45689655,
        0.47413793,
        0.49137931,
        0.50862069,
        0.52586207,
        0.54310345,
        0.56034483,
        0.57758621,
        0.59482759,
        0.61206897,
        0.62931034,
        0.64655172,
        0.6637931,
        0.68103448,
        0.69827586,
        0.71551724,
        0.73275862,
        0.75,
    ],
    "k": 2,
    "prob_size": 500,
}

logger = logging.getLogger(__name__)


def _optimize_group_lambdas(
    ground_truth: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    epsilon: float,
    gamma: list[float],
    k: int,
    prob_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    ground_truth = ground_truth.cpu()
    lower = lower.cpu()
    upper = upper.cpu()
    if ground_truth.ndim > 3:
        ground_truth = ground_truth.flatten(0, -3)
        lower = lower.flatten(0, -3)
        upper = upper.flatten(0, -3)

    mse = torch.mean((ground_truth - (lower + upper) / 2) ** 2, dim=0)
    thresholds = torch.quantile(
        mse.view(-1),
        torch.arange(0, 1, 1 / k, device=mse.device)[1:],
    ).unique()
    k = len(thresholds) + 1
    membership = torch.nn.functional.one_hot(
        torch.bucketize(mse, thresholds, right=False).squeeze(),
        num_classes=k,
    ).float()
    group_sizes = membership.view(-1, k).sum(dim=0).cpu().numpy()
    sampled_group_sizes = np.round(
        prob_size / np.prod(ground_truth.size()[-2:]) * group_sizes
    ).astype(int)
    logger.info(
        f"Initializing optimization problem with prob_size={np.sum(sampled_group_sizes):,}, "
        f"prob_nk={sampled_group_sizes}"
    )

    sampled_coordinates: tuple[list[int], list[int]] = ([], [])
    sampled_groups = []
    for group, group_sample_size in enumerate(sampled_group_sizes):
        group_coordinates = torch.nonzero(
            membership[:, :, group] == 1,
            as_tuple=True,
        )
        coordinate_indices = rng.choice(
            len(group_coordinates[0]),
            size=group_sample_size,
            replace=False,
        )
        for coordinates, group_coordinates_axis in zip(
            sampled_coordinates, group_coordinates
        ):
            coordinates.extend(group_coordinates_axis[coordinate_indices].tolist())
        sampled_groups.extend(group_sample_size * [group])

    sampled_coordinates_tensor = tuple(
        torch.tensor(coordinates) for coordinates in sampled_coordinates
    )
    center = (lower + upper) / 2
    sampled_interval = (upper - lower)[
        :,
        sampled_coordinates_tensor[0],
        sampled_coordinates_tensor[1],
    ].numpy()
    sampled_offset = torch.abs(ground_truth - center)[
        :,
        sampled_coordinates_tensor[0],
        sampled_coordinates_tensor[1],
    ].numpy()

    lambda_variable = cp.Variable(k)
    lambda_variable.value = 0.5 * np.ones(k)
    q = cp.Parameter(nonneg=True)
    interval_lambda = sampled_interval + 2 * lambda_variable[[sampled_groups]]
    risk = cp.sum(
        cp.pos(
            2 * (1 + q) * cp.multiply(cp.inv_pos(interval_lambda), sampled_offset) - q
        )
    ) / (ground_truth.size(0) * np.sum(sampled_group_sizes))
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(sampled_group_sizes, lambda_variable))),
        [
            lambda_variable >= 0,
            lambda_variable <= 0.5,
            risk <= epsilon,
        ],
    )

    solutions = []
    for value in tqdm(gamma, desc="Solving for gamma values"):
        q.value = value / (1 - value)
        problem.solve(verbose=False, warm_start=True)
        solutions.append(
            (
                torch.tensor(
                    lambda_variable.value,
                    dtype=torch.float,
                    device=device,
                ),
                problem.value,
            )
        )
    return membership, min(solutions, key=lambda solution: solution[1])[0]
