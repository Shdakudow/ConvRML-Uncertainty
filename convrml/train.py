"""Portable fresh training and epoch-boundary resume using recorded profiles.

No cluster submission or calibration is performed. Historical full-state resume
uses the archived original engine; these checkpoints have a separate schema.
"""

import argparse
import fcntl
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from .checkpoints import read_checkpoint, validate_identity
from .metrics import reconstruction_metrics
from .protocol import (
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
    parser.add_argument("--member", type=int)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.inference_mode()
def validate(model, method, dataset, run, device, workers, smoke):
    model.eval()
    totals, count = {}, 0
    for index, batch in enumerate(
        DataLoader(dataset, batch_size=run["microbatch"], num_workers=workers)
    ):
        if smoke and index == 2:
            break
        y, x = batch[:, 0].to(device), batch[:, 1].to(device)
        with autocast(run, device):
            prediction = center(model, method, y).float().clamp(0, 1)
        for key, values in reconstruction_metrics(prediction, x).items():
            totals[key] = totals.get(key, 0.0) + float(values.sum())
        count += len(batch)
    return dict(count=count, clipped=True, **{k: v / count for k, v in totals.items()})


def run_training(args, *, factory=None):
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
            checkpoint, _ = read_checkpoint(latest)
            if checkpoint.get("schema") != run["schema"]:
                raise ValueError(
                    "Historical optimizer/RNG states require the original engine"
                )
            if (
                checkpoint["run"] != run
                or checkpoint["data"] != data
                or checkpoint["smoke"] != args.smoke
            ):
                raise ValueError("Resume configuration or dataset identity changed")
            if (
                checkpoint["device"] != str(device)
                or checkpoint["workers"] != args.workers
            ):
                raise ValueError(
                    "Exact portable resume retains device and worker settings"
                )
            if checkpoint["source_sha256"] != source:
                raise ValueError(
                    "Portable training source changed; use its original release to resume"
                )
            if checkpoint["environment"] != software:
                raise ValueError(
                    "Portable resume requires the same recorded software versions"
                )
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            completed, global_step, history = (
                checkpoint["epoch"],
                checkpoint["global_step"],
                checkpoint["history"],
            )
            parent = checkpoint.get("parent")
            torch.set_rng_state(checkpoint["rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            quantile_rng.set_state(checkpoint["quantile_rng"])
            data_rng.set_state(checkpoint["data_rng"])
        elif latest.exists():
            raise FileExistsError(
                "Output already contains latest.pt; use --resume or a new output"
            )
        elif args.initialize:
            checkpoint, digest = read_checkpoint(args.initialize)
            info = validate_identity(
                checkpoint, profile(3, "raw_clean", args.method), smoke=args.smoke
            )
            if (info["method"], info["dataset"], info["epoch"]) != (
                args.method,
                "raw_clean",
                30,
            ):
                raise ValueError(
                    "Adaptation parent must be this method's clean raw epoch 30"
                )
            if info["smoke"] and not args.smoke:
                raise ValueError("Cannot initialize a full run from smoke weights")
            model.load_state_dict(checkpoint["model"], strict=True)
            parent = dict(sha256=digest, **info)
        if completed > target:
            raise ValueError("Target precedes the completed epoch")
        if completed and completed % settings()["checkpoint_interval"] == 0:
            destination = out / "checkpoints" / f"epoch_{completed:03d}.pt"
            if not destination.exists():
                save_checkpoint(
                    destination,
                    {
                        key: checkpoint[key]
                        for key in (
                            "schema",
                            "run",
                            "data",
                            "model",
                            "epoch",
                            "history",
                            "smoke",
                            "parent",
                            "source_sha256",
                            "environment",
                        )
                    },
                )
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
            if run["shuffle"] == "per_epoch_permutation":
                seed = run["seeds"]["data"] + epoch
                order = torch.randperm(
                    len(training), generator=torch.Generator().manual_seed(seed)
                ).tolist()
                loader = DataLoader(
                    Subset(training, order),
                    batch_size=run["batch_size"],
                    num_workers=args.workers,
                    generator=torch.Generator().manual_seed(seed + 10000),
                )
            else:
                loader = DataLoader(
                    training,
                    batch_size=run["batch_size"],
                    shuffle=True,
                    generator=data_rng,
                    num_workers=args.workers,
                )
            model.train()
            total, count, started = 0.0, 0, time.monotonic()
            for index, batch in enumerate(loader):
                if args.smoke and index == 2:
                    break
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(global_step, horizon, run)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                batch_loss = 0.0
                for sub in batch.split(run["microbatch"]):
                    y, x = sub[:, 0].to(device), sub[:, 1].to(device)
                    with autocast(run, device):
                        loss = (
                            model.loss(y, x, generator=quantile_rng)
                            if args.method == "qutcc"
                            else model.loss(y, x)
                        )
                        loss = loss * len(sub) / len(batch)
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
                total += batch_loss * len(batch)
                count += len(batch)
                global_step += 1
            training_seconds = time.monotonic() - started
            measured = validate(
                model, args.method, validation, run, device, args.workers, args.smoke
            )
            seconds = time.monotonic() - started
            row = dict(
                epoch=epoch,
                loss=total / count,
                lr=lr,
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
