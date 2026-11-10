"""Train, validate and save one of the four reconstruction/uncertainty models.

Read run_training for the workflow, setup_training for initialization, and
train_batch for one optimizer update. Profiles remain JSON-compatible dictionaries.
Losses live in models/. Calibration and final test evaluation are separate.
"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset, Subset

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
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON file listing train/val image IDs and, for DLMD, image paths.",
    )
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


@dataclass
class TrainingState:
    """Live training objects and progress; checkpoint serialization stays explicit."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    config: dict
    training: Dataset
    validation: Dataset
    output: Path
    identity: dict
    quantile_rng: torch.Generator
    data_rng: torch.Generator
    schedule_steps: int
    completed_epoch: int
    global_step: int
    history: list
    parent: object


def run_training(args, *, factory=None):
    """Set up a run, then train, validate and save each completed epoch."""
    config, target_epoch = training_config(args)
    state = setup_training(args, config, target_epoch, factory=factory)
    for epoch in range(state.completed_epoch + 1, target_epoch + 1):
        loader = create_train_loader(state, epoch, args.workers)
        stats = train_one_epoch(
            state.model,
            state.optimizer,
            loader,
            run=config,
            device=state.device,
            quantile_rng=state.quantile_rng,
            global_step=state.global_step,
            schedule_steps=state.schedule_steps,
            smoke=args.smoke,
        )
        state.global_step = stats["global_step"]
        validation_metrics = validate(
            state.model,
            args.method,
            state.validation,
            config,
            state.device,
            args.workers,
            args.smoke,
        )
        save_epoch(state, epoch, stats, validation_metrics)
        print(
            f"Epoch {epoch}/{target_epoch}: loss={stats['loss']:.6g}, validation SSIM={validation_metrics['ssim']:.5f}",
            flush=True,
        )
    return state.output / "latest.pt"


def training_config(args):
    """Validate the requested stopping point without changing the LR horizon."""
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

    return config, target_epoch


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


def setup_training(args, config, target_epoch, *, factory=None):
    """Open the fixed splits, create seeded objects, and restore state if requested."""
    # Load only train/validation; calibration and test pairs are not used here.
    # The split file lists image identities/paths; the datasets load pixels.
    # Keep the leakage check separate from image loading.
    data_identity = audit_manifest(args.manifest)
    if not {"train", "val"} <= data_identity["splits"].keys():
        raise ValueError("Training requires train and val splits")
    training = dataset_for(args.dataset, args.manifest, args.pairs, "train")
    validation = dataset_for(args.dataset, args.manifest, args.pairs, "val")
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    if latest.exists() and not args.resume:
        raise FileExistsError("Use --resume or a new output folder")

    # make_model also seeds itself for standalone evaluation. Seed here as well
    # so injected factories follow the same initialization contract.
    torch.manual_seed(config["seeds"]["initialization"])
    model = (factory or make_model)(config).to(device)
    steps_per_epoch = math.ceil(len(training) / config["batch_size"])
    schedule_steps = config["epochs"] * steps_per_epoch
    optimizer = create_optimizer(model, config, schedule_steps)
    # Reset the training stream after initialization, preserving dropout draws.
    torch.manual_seed(config["seeds"]["training"])
    quantile_rng = torch.Generator(device=device).manual_seed(
        config["seeds"]["quantile"]
    )
    data_rng = torch.Generator().manual_seed(config["seeds"]["data"])
    identity = dict(
        run=config,
        data=data_identity,
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

    return TrainingState(
        model=model,
        optimizer=optimizer,
        device=device,
        config=config,
        training=training,
        validation=validation,
        output=output,
        identity=identity,
        quantile_rng=quantile_rng,
        data_rng=data_rng,
        schedule_steps=schedule_steps,
        completed_epoch=completed_epoch,
        global_step=global_step,
        history=history,
        parent=parent,
    )


def create_train_loader(state, epoch, workers):
    """Preserve the profile's epoch permutation or persistent shuffle stream."""
    config = state.config
    if config["shuffle"] == "per_epoch_permutation":
        seed = config["seeds"]["data"] + epoch
        order = torch.randperm(
            len(state.training), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        return DataLoader(
            Subset(state.training, order),
            batch_size=config["batch_size"],
            num_workers=workers,
            generator=torch.Generator().manual_seed(seed + 10000),
        )
    return DataLoader(
        state.training,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=state.data_rng,
        num_workers=workers,
    )


def save_epoch(state, epoch, stats, validation_metrics):
    """Record validation/timing and save recovery state plus scheduled milestones."""
    seconds = time.monotonic() - stats["started"]
    state.history.append(
        dict(
            epoch=epoch,
            loss=stats["loss"],
            lr=stats["lr"],
            seconds=seconds,
            training_seconds=stats["training_seconds"],
            validation_seconds=seconds - stats["training_seconds"],
            validation=validation_metrics,
        )
    )
    checkpoint = dict(
        state.identity,
        schema=state.config["schema"],
        model=state.model.state_dict(),
        epoch=epoch,
        history=state.history,
        parent=state.parent,
        global_step=state.global_step,
    )
    save_training_progress(
        state.output,
        checkpoint,
        state.optimizer,
        state.device,
        state.quantile_rng,
        state.data_rng,
    )
    state.completed_epoch = epoch


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


def training_loss(model, measurement, target, run, quantile_rng):
    """Only QUTCC consumes the dedicated random-quantile stream."""
    if run["method"] == "qutcc":
        return model.loss(measurement, target, generator=quantile_rng)
    return model.loss(measurement, target)


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
            loss = training_loss(model, measurement, target, run, quantile_rng)
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
