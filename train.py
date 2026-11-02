"""Train, validate and save one of the four reconstruction/uncertainty models.

Run run_training for the complete workflow; train_one_epoch shows each update.
Losses live in models/. Calibration and final test evaluation are separate.
"""

import argparse
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from convrml.checkpoints import (
    resume_training,
    initialize_from_clean,
    save_training_progress,
)
from convrml.metrics import reconstruction_metrics
from convrml.protocol import (
    audit_manifest,
    autocast,
    center,
    dataset_for,
    learning_rate,
    make_model,
    profile,
    settings,
    write_json,
)


def arguments(argv=None):
    """Read command-line options; the selected dataset supplies its saved profile."""
    datasets = settings()["datasets"]
    methods = ["im2im", "ensemble", "mc_dropout", "qutcc"]
    parser = argparse.ArgumentParser(description=__doc__)

    # Model and data.
    parser.add_argument("--dataset", choices=list(datasets), required=True)
    parser.add_argument("--method", choices=methods, required=True)
    parser.add_argument("--member", type=int, help="Ensemble member index, 0-9.")
    parser.add_argument("--manifest", type=Path, required=True, help="Train/val JSON.")
    parser.add_argument("--pairs", type=Path, help="PLD NumPy cache; omit for DLMD.")
    parser.add_argument("--output", type=Path, required=True, help="Run output folder.")

    # Training duration and starting point.
    parser.add_argument("--epochs", type=int, help="Stop at this absolute epoch.")
    parser.add_argument("--resume", action="store_true", help="Resume latest.pt.")
    parser.add_argument(
        "--initialize", type=Path, help="Clean epoch-30 adaptation weights."
    )

    # Runtime and small fixture checks.
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--smoke", action="store_true", help="At most two batches, one epoch."
    )

    args = parser.parse_args(argv)
    args.experiment = datasets[args.dataset]["experiment"]
    return args


def run_training(args, *, factory=None):
    """Load pairs -> create Adam/model -> train/validate epochs -> save state."""
    config = profile(args.experiment, args.dataset, args.method, args.member)
    target_epoch = args.epochs if args.epochs is not None else config["epochs"]
    if args.smoke:
        target_epoch = 1
    if not 1 <= target_epoch <= config["epochs"] or args.workers < 0:
        raise ValueError("Invalid epoch or worker count")
    if args.resume and args.initialize:
        raise ValueError("Choose resume or initialization, not both")
    if config["family"] == "adaptation" and not (args.initialize or args.resume):
        raise ValueError("Adaptation requires clean weights or --resume")
    if config["family"] != "adaptation" and args.initialize:
        raise ValueError("Clean comparisons start from scratch")

    # Load only train/validation; calibration and test pairs are not used here.
    data = audit_manifest(args.manifest)
    if not {"train", "val"} <= data["splits"].keys():
        raise ValueError("Training requires train and val splits")
    training = dataset_for(args.dataset, args.manifest, args.pairs, "train")
    validation = dataset_for(args.dataset, args.manifest, args.pairs, "val")
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    if latest.exists() and not args.resume:
        raise FileExistsError("Use --resume or a new output folder")

    model = (factory or make_model)(config).to(device)
    schedule_steps = config["epochs"] * math.ceil(len(training) / config["batch_size"])
    initial_lr = learning_rate(0, schedule_steps, config)
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=initial_lr,
            betas=tuple(config["betas"]),
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            betas=tuple(config["betas"]),
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError("Expected Adam or AdamW")
    torch.manual_seed(config["seeds"]["training"])
    quantile_rng = torch.Generator(device=device).manual_seed(
        config["seeds"]["quantile"]
    )
    data_rng = torch.Generator().manual_seed(config["seeds"]["data"])
    identity = dict(
        run=config,
        data=data,
        smoke=args.smoke,
        device=str(device),
        workers=args.workers,
    )
    completed_epoch = global_step = 0
    history, parent = [], None
    if args.resume:
        state = resume_training(
            latest,
            model,
            optimizer,
            identity,
            device=device,
            quantile_rng=quantile_rng,
            data_rng=data_rng,
        )
        completed_epoch, global_step = state["epoch"], state["global_step"]
        history, parent = state["history"], state.get("parent")
    elif args.initialize:
        parent = initialize_from_clean(
            model, args.initialize, args.method, smoke=args.smoke
        )
    if completed_epoch > target_epoch:
        raise ValueError("Target precedes the saved epoch")
    write_json(output / "config.json", dict(identity, parent=parent))

    for epoch in range(completed_epoch + 1, target_epoch + 1):
        # Retain each profile's recorded data order and ensemble member seed.
        if config["shuffle"] == "per_epoch_permutation":
            seed = config["seeds"]["data"] + epoch
            order = torch.randperm(
                len(training), generator=torch.Generator().manual_seed(seed)
            ).tolist()
            loader = DataLoader(
                Subset(training, order),
                batch_size=config["batch_size"],
                num_workers=args.workers,
                generator=torch.Generator().manual_seed(seed + 10000),
            )
        else:
            loader = DataLoader(
                training,
                batch_size=config["batch_size"],
                shuffle=True,
                generator=data_rng,
                num_workers=args.workers,
            )
        stats = train_one_epoch(
            model,
            optimizer,
            loader,
            run=config,
            device=device,
            quantile_rng=quantile_rng,
            global_step=global_step,
            schedule_steps=schedule_steps,
            smoke=args.smoke,
        )
        global_step = stats["global_step"]
        measured = validate(
            model, args.method, validation, config, device, args.workers, args.smoke
        )
        seconds = time.monotonic() - stats["started"]
        history.append(
            dict(
                epoch=epoch,
                loss=stats["loss"],
                lr=stats["lr"],
                seconds=seconds,
                training_seconds=stats["training_seconds"],
                validation_seconds=seconds - stats["training_seconds"],
                validation=measured,
            )
        )
        state = dict(
            identity,
            schema=config["schema"],
            model=model.state_dict(),
            epoch=epoch,
            history=history,
            parent=parent,
            global_step=global_step,
        )
        save_training_progress(output, state, optimizer, device, quantile_rng, data_rng)
        print(
            f"Epoch {epoch}/{target_epoch}: loss={stats['loss']:.6g}, validation SSIM={measured['ssim']:.5f}",
            flush=True,
        )
    return latest


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
                if run["method"] == "qutcc":
                    loss = model.loss(measurement, target, generator=quantile_rng)
                else:
                    loss = model.loss(measurement, target)
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


@torch.inference_mode()
def validate(model, method, dataset, run, device, workers, smoke):
    """Measure mean clipped PSNR/SSIM/MAE without changing model weights.

    Validation measures reconstruction only. MC dropout is deterministic here;
    ensemble training validates one member; QUTCC predicts at q=0.5. Final
    stochastic/ensemble interval inference and calibration happen separately.
    """
    model.eval()
    totals = {}
    image_count = 0
    loader = DataLoader(dataset, batch_size=run["microbatch"], num_workers=workers)
    for batch_index, batch in enumerate(loader):
        if smoke and batch_index == 2:
            break
        measurement = batch[:, 0].to(device)
        target = batch[:, 1].to(device)
        with autocast(run, device):
            prediction = center(model, method, measurement).float()
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite validation prediction")
        prediction = prediction.clamp(0, 1)
        for name, values in reconstruction_metrics(prediction, target).items():
            totals[name] = totals.get(name, 0.0) + float(values.sum())
        image_count += len(batch)
    means = {name: total / image_count for name, total in totals.items()}
    return dict(count=image_count, clipped=True, **means)


if __name__ == "__main__":
    run_training(arguments())
