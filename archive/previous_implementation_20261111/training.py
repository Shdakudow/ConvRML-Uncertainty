"""Training settings, AdamW, learning-rate schedule and shared optimization loops.

Models define their own loss. This module accumulates gradients, updates AdamW,
and records epoch statistics; it does not select a loss based on model type.
Only QUTCC uses the supplied generator to sample training quantiles.
"""

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import torch

SETTINGS_PATH = Path(__file__).resolve().parent / "config.json"


def settings():
    """Read recorded settings as a fresh dictionary, without changing the JSON file."""
    with SETTINGS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def profile(experiment, dataset, method, member=None):
    """Build the run dictionary consumed by training, checkpoint loading and inference.

    Dataset/method select a recorded optimizer profile and shuffle policy.
    Separate seeds control initialization, training randomness, data order and q
    draws. ``member`` is required only for an individual ensemble member (0-9).
    """
    config = settings()
    data = config["datasets"][dataset]
    if data["experiment"] != experiment:
        raise ValueError("Dataset does not belong to the requested experiment")
    if method not in config["methods"][f"experiment{experiment}"]:
        raise ValueError("Method is not part of this experiment")
    if method == "ensemble" and member is None:
        raise ValueError("Choose --member 0..9 for ensemble training/loading")
    if method != "ensemble" and member is not None:
        raise ValueError("Only ensemble models use --member")
    if member is not None and member not in range(10):
        raise ValueError("Ensemble member index must be 0..9")
    # Select a recorded optimizer profile, not a different model architecture.
    if dataset == "raw_corrupted":
        family = "adaptation"
    elif method == "qutcc" or experiment == 3:
        family = "paper"
    else:
        family = "baseline"
    result = dict(config["profiles"][family])
    if data["loader"] == "images":
        result["microbatch"] = 1
    # Ensemble members use independent initialization AND image-order streams.
    policy = config["seed_policy"]
    if family == "baseline":
        index = member if method == "ensemble" else policy["arm_index"][method]
        seeds = {
            name: policy["baseline_" + name] + index
            for name in ("initialization", "training", "data", "quantile")
        }
        shuffle = "persistent_generator"
    else:
        seed = policy["paper_base"] + data["camera_index"]
        seeds = dict(
            initialization=seed, training=seed, data=seed, quantile=seed + 1000
        )
        shuffle = "per_epoch_permutation"
    return dict(
        result,
        schema=config["schema"],
        experiment=experiment,
        dataset=dataset,
        method=method,
        member=member,
        seeds=seeds,
        shuffle=shuffle,
        dropout=config["dropout_probability"],
        family=family,
    )


def autocast(run, device):
    """Use the recorded BF16 context, or preserve normal FP32 computation."""
    if run["precision"] == "bfloat16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


def learning_rate(step, total_steps, run):
    """Recorded constant or warmup/cosine schedule; resume keeps its full horizon."""
    peak = run["peak_lr"]
    if run["schedule"] == "constant":
        return peak
    # The saved profile warms up for 30% of all optimizer updates, then decays
    # over the remaining 70%. --epochs changes when to stop, not this horizon.
    progress = step / total_steps
    initial_lr = peak / 10
    warmup_fraction = 0.3
    if progress <= warmup_fraction:
        return initial_lr + progress / warmup_fraction * (peak - initial_lr)

    minimum_lr = initial_lr / 1e4
    # Keep the recorded operation order for numerical parity with saved runs.
    cosine_angle = math.pi * (progress - warmup_fraction) / 0.7
    cosine = math.cos(cosine_angle)
    return minimum_lr + (peak - minimum_lr) / 2 * (cosine + 1)


def create_optimizer(model, config, schedule_steps):
    """Use AdamW for new runs, with explicit profile hyperparameters."""
    if config["optimizer"] != "AdamW":
        raise ValueError(
            "New runs use AdamW; resume historical Adam runs with their original code"
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate(0, schedule_steps, config),
        betas=tuple(config["betas"]),
        weight_decay=config["weight_decay"],
    )


def train_one_epoch(
    model,
    optimizer,
    loader,
    *,
    run,
    device,
    quantile_rng,
    global_step,
    schedule_steps,
    smoke=False,
):
    """Learn from one pass through the training pairs and return epoch statistics.

    Each batch is (B,2,3,H,W): measurement first, target second. Microbatches
    accumulate gradients for one effective batch, followed by one optimizer step.
    The model's loss method selects MSE, Gaussian NLL or pinball as appropriate.
    """
    model.train()
    total_loss = 0.0
    image_count = 0
    started = time.monotonic()
    for batch_index, batch in enumerate(loader):
        if smoke and batch_index == 2:
            break

        batch_loss, current_lr = train_batch(
            model,
            optimizer,
            batch,
            run=run,
            device=device,
            quantile_rng=quantile_rng,
            global_step=global_step,
            schedule_steps=schedule_steps,
        )
        global_step += 1
        total_loss += batch_loss * len(batch)
        image_count += len(batch)

    return dict(
        loss=total_loss / image_count,
        lr=current_lr,
        global_step=global_step,
        training_seconds=time.monotonic() - started,
        started=started,
    )


def train_batch(
    model,
    optimizer,
    batch,
    *,
    run,
    device,
    quantile_rng,
    global_step,
    schedule_steps,
):
    """Take one optimizer step, weighting microbatches by their image counts."""
    # Set this update's learning rate and clear the preceding gradients.
    optimizer.zero_grad(set_to_none=True)
    current_lr = learning_rate(global_step, schedule_steps, run)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = current_lr

    # Compute gradients. Weighting keeps a short final microbatch correct.
    batch_loss = 0.0
    for microbatch in batch.split(run["microbatch"]):
        measurement = microbatch[:, 0].to(device)
        target = microbatch[:, 1].to(device)
        with autocast(run, device):
            loss = model.loss(measurement, target, generator=quantile_rng)
            loss = loss * len(microbatch) / len(batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        batch_loss += float(loss.detach())

    # Limit gradient norm only for profiles that specify it, then learn.
    if run["gradient_clip"] is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), run["gradient_clip"], error_if_nonfinite=True
        )
    optimizer.step()
    return batch_loss, current_lr
