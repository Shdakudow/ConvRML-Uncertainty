"""Common training workflow: load data, train an epoch, validate and checkpoint.

Portable checkpoints retain optimizer and random-generator state for resume.
Historical full-state resumes use their archived engine and checkpoint schema.
"""

import argparse
import fcntl
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from convrml.checkpoints import (
    save_checkpoint,
    resume_training,
    initialize_from_clean,
    model_checkpoint,
)
from convrml.metrics import reconstruction_metrics
from convrml.protocol import (
    audit_manifest,
    autocast,
    center,
    dataset_for,
    environment,
    learning_rate,
    make_model,
    profile,
    settings,
    write_json,
    check_shape,
    source_fingerprint,
    hardware,
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=int, choices=[2, 3], help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dataset", choices=list(settings()["datasets"]), required=True
    )
    parser.add_argument(
        "--method", choices=["im2im", "ensemble", "mc_dropout", "qutcc"], required=True
    )
    parser.add_argument(
        "--member",
        type=int,
        help="Ensemble member index 0-9; one output folder per member",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON with train/val IDs; DLMD rows also contain image paths",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        help="PLD uint8 NumPy pair cache; omit for DLMD image-file data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Run folder for config, history, latest state and five-epoch weights",
    )
    parser.add_argument(
        "--epochs", type=int, help="Absolute target epoch; schedule horizon stays fixed"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume this portable run's latest.pt"
    )
    parser.add_argument(
        "--initialize",
        type=Path,
        help="Clean epoch-30 parent for corruption adaptation",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two batches, one epoch; explicitly marked, small data allowed",
    )
    args = parser.parse_args(argv)
    if args.experiment is None:
        args.experiment = settings()["datasets"][args.dataset]["experiment"]
    return args


@torch.inference_mode()
def validate(model, method, dataset, run, device, workers, smoke):
    """Return count and mean clipped PSNR/SSIM/MAE using one prediction per image.

    This validation step measures reconstruction quality only. Interval calibration
    and held-out test evaluation happen separately in calibration.py.
    """
    model.eval()
    totals, count = {}, 0
    for index, batch in enumerate(
        DataLoader(dataset, batch_size=run["microbatch"], num_workers=workers)
    ):
        if smoke and index == 2:
            break
        measurement, target = batch[:, 0].to(device), batch[:, 1].to(device)
        with autocast(run, device):
            prediction = center(model, method, measurement).float().clamp(0, 1)
        for key, values in reconstruction_metrics(prediction, target).items():
            totals[key] = totals.get(key, 0.0) + float(values.sum())
        count += len(batch)
    return dict(count=count, clipped=True, **{k: v / count for k, v in totals.items()})


def make_training_loader(dataset, run, epoch, workers, data_rng):
    """Keep the recorded shuffle policy and independent ensemble data streams."""
    if run["shuffle"] == "per_epoch_permutation":
        seed = run["seeds"]["data"] + epoch
        order = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        return DataLoader(
            Subset(dataset, order),
            batch_size=run["batch_size"],
            num_workers=workers,
            generator=torch.Generator().manual_seed(seed + 10000),
        )
    return DataLoader(
        dataset,
        batch_size=run["batch_size"],
        shuffle=True,
        generator=data_rng,
        num_workers=workers,
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
    """Forward, loss, backward and optimizer step, with microbatch accumulation.

    Loader batches are (B,2,3,H,W) in [0,1]. Each model returns a scalar loss;
    weighting by microbatch size preserves the effective batch's mean objective.
    """
    model.train()
    total_loss, count, started = 0.0, 0, time.monotonic()
    for index, batch in enumerate(loader):
        if smoke and index == 2:
            break
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(global_step, schedule_steps, run)
        for group in optimizer.param_groups:
            group["lr"] = lr
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
        if run["gradient_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                run["gradient_clip"],
                error_if_nonfinite=True,
            )
        optimizer.step()
        total_loss += batch_loss * len(batch)
        count += len(batch)
        global_step += 1
    return dict(
        loss=total_loss / count,
        lr=lr,
        global_step=global_step,
        training_seconds=time.monotonic() - started,
        started=started,
    )


def run_training(args, *, factory=None):
    """Train to args.epochs (absolute), save each completed epoch, and return latest.pt.

    ``args`` comes from arguments(); ``factory`` optionally supplies a small model
    for workflow tests. Production runs use the shared make_model constructor.
    """
    run = profile(args.experiment, args.dataset, args.method, args.member)
    target = args.epochs if args.epochs is not None else run["epochs"]
    if args.smoke:
        if args.epochs not in (None, 1):
            raise ValueError("Smoke mode has exactly one target epoch")
        target = 1
    if not 1 <= target <= run["epochs"] or args.workers < 0:
        raise ValueError("Invalid target epoch or worker count")
    if args.resume and args.initialize:
        raise ValueError("Choose resume or clean-parent initialization, not both")
    if run["family"] == "adaptation" and not (args.initialize or args.resume):
        raise ValueError("Corruption adaptation requires its clean epoch-30 parent")
    if run["family"] != "adaptation" and args.initialize:
        raise ValueError("Clean comparisons start from scratch")
    data = audit_manifest(args.manifest, args.dataset, smoke=args.smoke)
    if not {"train", "val"} <= data["splits"].keys():
        raise ValueError("Training requires train and validation splits")
    training = dataset_for(args.dataset, args.manifest, args.pairs, "train")
    validation = dataset_for(args.dataset, args.manifest, args.pairs, "val")
    check_shape(training, args.dataset, smoke=args.smoke)
    check_shape(validation, args.dataset, smoke=args.smoke)
    device = torch.device(args.device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"
    history, completed, global_step, parent = [], 0, 0, None
    source = source_fingerprint()
    software = environment()
    with (out / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = (factory or make_model)(run).to(device)
        optimizer_type = getattr(torch.optim, run["optimizer"])
        optimizer = optimizer_type(
            model.parameters(),
            lr=run["peak_lr"],
            betas=tuple(run["betas"]),
            weight_decay=run["weight_decay"],
        )
        torch.manual_seed(run["seeds"]["training"])
        quantile_rng = torch.Generator(device=device).manual_seed(
            run["seeds"]["quantile"]
        )
        data_rng = torch.Generator().manual_seed(run["seeds"]["data"])
        if args.resume:
            expected = dict(
                run=run,
                data=data,
                smoke=args.smoke,
                device=str(device),
                workers=args.workers,
                source_sha256=source,
                environment=software,
            )
            checkpoint = resume_training(
                latest,
                model,
                optimizer,
                expected,
                device=device,
                quantile_rng=quantile_rng,
                data_rng=data_rng,
            )
            completed = checkpoint["epoch"]
            global_step = checkpoint["global_step"]
            history = checkpoint["history"]
            parent = checkpoint.get("parent")
        elif latest.exists():
            raise FileExistsError(
                "Output already contains latest.pt; use --resume or a new output"
            )
        elif args.initialize:
            parent = initialize_from_clean(
                model, args.initialize, args.method, smoke=args.smoke
            )
        if completed > target:
            raise ValueError("Target precedes the completed epoch")
        if completed and completed % settings()["checkpoint_interval"] == 0:
            destination = out / "checkpoints" / f"epoch_{completed:03d}.pt"
            if not destination.exists():
                save_checkpoint(destination, model_checkpoint(checkpoint))
        write_json(
            out / "config.json",
            dict(
                run=run,
                data=data,
                smoke=args.smoke,
                parent=parent,
                environment=software,
                hardware=hardware(device),
                device=str(device),
                workers=args.workers,
                source_sha256=source,
            ),
        )
        horizon = run["epochs"] * math.ceil(len(training) / run["batch_size"])
        for epoch in range(completed + 1, target + 1):
            loader = make_training_loader(training, run, epoch, args.workers, data_rng)
            epoch_stats = train_one_epoch(
                model,
                optimizer,
                loader,
                run=run,
                device=device,
                quantile_rng=quantile_rng,
                global_step=global_step,
                schedule_steps=horizon,
                smoke=args.smoke,
            )
            global_step = epoch_stats["global_step"]
            training_seconds = epoch_stats["training_seconds"]
            measured = validate(
                model, args.method, validation, run, device, args.workers, args.smoke
            )
            seconds = time.monotonic() - epoch_stats["started"]
            row = dict(
                epoch=epoch,
                loss=epoch_stats["loss"],
                lr=epoch_stats["lr"],
                seconds=seconds,
                training_seconds=training_seconds,
                validation_seconds=seconds - training_seconds,
                validation=measured,
            )
            history.append(row)
            compact = dict(
                schema=run["schema"],
                run=run,
                data=data,
                model=model.state_dict(),
                epoch=epoch,
                history=history,
                smoke=args.smoke,
                parent=parent,
                source_sha256=source,
                environment=software,
            )
            full = dict(
                compact,
                optimizer=optimizer.state_dict(),
                global_step=global_step,
                rng=torch.get_rng_state(),
                cuda_rng=(
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                ),
                quantile_rng=quantile_rng.get_state(),
                data_rng=data_rng.get_state(),
                device=str(device),
                workers=args.workers,
            )
            save_checkpoint(latest, full)
            if epoch % settings()["checkpoint_interval"] == 0:
                destination = out / "checkpoints" / f"epoch_{epoch:03d}.pt"
                if destination.exists():
                    raise FileExistsError(destination)
                save_checkpoint(destination, compact)
            write_json(out / "history.json", history)
            print(
                f"Epoch {epoch}/{target}: loss={row['loss']:.6g}, validation SSIM={row['validation']['ssim']:.5f}",
                flush=True,
            )
    return latest


def main():
    run_training(arguments())


if __name__ == "__main__":
    main()
