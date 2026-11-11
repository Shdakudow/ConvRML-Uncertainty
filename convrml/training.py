"""Shared optimization loop for all four reconstruction models.

Models define their own loss. This module accumulates gradients, updates AdamW,
and records epoch statistics; it does not select a loss based on model type.
Only QUTCC uses the supplied generator to sample training quantiles.
"""

import time

import torch

from .protocol import autocast, learning_rate


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
