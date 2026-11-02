"""Train one shared reconstruction/uncertainty model on paired measurements.

Start with run_training for the complete workflow, then train_one_epoch for the
actual learning loop: zero gradients -> predict/loss -> backward -> Adam step.
Model definitions and losses live in models/; recorded settings live in configs/.
Checkpoint serialization and resume checks live in convrml/checkpoints.py.
Calibration and final test evaluation are separate commands in calibration.py.
"""

import argparse
import fcntl
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from convrml.checkpoints import (
    save_checkpoint,  # Retained for the existing train.save_checkpoint API.
    resume_training,
    initialize_from_clean,
    save_training_progress,
    restore_missing_milestone,
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


def run_training(args, *, factory=None):
    """Load pairs, train/validate each epoch, and return the latest checkpoint path.

    args.epochs is the absolute stopping epoch. The learning-rate schedule always
    uses the full recorded horizon, so stopping/resuming does not restart warmup.
    ``factory`` is an optional tiny-model constructor used only by fixture tests.
    """
    # 1. Read the recorded settings and load only train/validation pairs.
    config = profile(args.experiment, args.dataset, args.method, args.member)
    target_epoch = validate_training_request(args, config)
    training, validation, data_identity = load_training_data(args)
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    identity = dict(
        run=config,
        data=data_identity,
        smoke=args.smoke,
        device=str(device),
        workers=args.workers,
        source_sha256=source_fingerprint(),
        environment=environment(),
    )
    steps_per_epoch = math.ceil(len(training) / config["batch_size"])
    schedule_steps = config["epochs"] * steps_per_epoch

    # One process owns a run folder; another writer must not replace its state.
    with (output / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        # 2. Create the model, Adam/AdamW, and independent random streams.
        model = (factory or make_model)(config).to(device)
        optimizer = create_optimizer(model, config, schedule_steps)
        torch.manual_seed(config["seeds"]["training"])
        quantile_rng = torch.Generator(device=device).manual_seed(
            config["seeds"]["quantile"]
        )
        data_rng = torch.Generator().manual_seed(config["seeds"]["data"])
        history = []
        completed_epoch = global_step = 0
        parent = None

        # 3. Resume all state, or initialize adaptation from clean weights.
        if args.resume:
            checkpoint = resume_training(
                latest,
                model,
                optimizer,
                identity,
                device=device,
                quantile_rng=quantile_rng,
                data_rng=data_rng,
            )
            completed_epoch = checkpoint["epoch"]
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
        if completed_epoch > target_epoch:
            raise ValueError("Target precedes the completed epoch")
        if args.resume:
            restore_missing_milestone(output, checkpoint)
        write_json(
            output / "config.json",
            dict(identity, parent=parent, hardware=hardware(device)),
        )

        # 4. Repeat the learning loop, validation, and checkpoint save.
        for epoch in range(completed_epoch + 1, target_epoch + 1):
            loader = make_training_loader(
                training, config, epoch, args.workers, data_rng
            )
            training_stats = train_one_epoch(
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
            global_step = training_stats["global_step"]
            validation_metrics = validate(
                model,
                args.method,
                validation,
                config,
                device,
                args.workers,
                args.smoke,
            )
            epoch_seconds = time.monotonic() - training_stats["started"]
            history.append(
                dict(
                    epoch=epoch,
                    loss=training_stats["loss"],
                    lr=training_stats["lr"],
                    seconds=epoch_seconds,
                    training_seconds=training_stats["training_seconds"],
                    validation_seconds=epoch_seconds
                    - training_stats["training_seconds"],
                    validation=validation_metrics,
                )
            )
            checkpoint = dict(
                identity,
                schema=config["schema"],
                model=model.state_dict(),
                epoch=epoch,
                history=history,
                parent=parent,
                global_step=global_step,
            )
            save_training_progress(
                output, checkpoint, optimizer, device, quantile_rng, data_rng
            )
            print(
                f"Epoch {epoch}/{target_epoch}: loss={training_stats['loss']:.6g}, "
                f"validation SSIM={validation_metrics['ssim']:.5f}",
                flush=True,
            )
    return latest


def create_optimizer(model, config, schedule_steps):
    """Choose Adam explicitly and start at the first scheduled learning rate.

    The paper profile starts at 1e-5, warms to 1e-4, then decays. The baseline
    AdamW and adaptation Adam profiles use their recorded constant rates.
    train_one_epoch updates optimizer.param_groups before every learning step.
    """
    initial_lr = learning_rate(0, schedule_steps, config)
    betas = tuple(config["betas"])
    weight_decay = config["weight_decay"]
    if config["optimizer"] == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=initial_lr, betas=betas, weight_decay=weight_decay
        )
    if config["optimizer"] == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=initial_lr, betas=betas, weight_decay=weight_decay
        )
    raise ValueError("Unsupported optimizer: " + config["optimizer"])


def validate_training_request(args, config):
    """Return the stopping epoch after checking fresh/resume/adaptation options."""
    target_epoch = args.epochs if args.epochs is not None else config["epochs"]
    if args.smoke:
        if args.epochs not in (None, 1):
            raise ValueError("Smoke mode has exactly one target epoch")
        target_epoch = 1
    if not 1 <= target_epoch <= config["epochs"] or args.workers < 0:
        raise ValueError("Invalid target epoch or worker count")
    if args.resume and args.initialize:
        raise ValueError("Choose resume or clean-parent initialization, not both")
    if config["family"] == "adaptation" and not (args.initialize or args.resume):
        raise ValueError("Corruption adaptation requires its clean epoch-30 parent")
    if config["family"] != "adaptation" and args.initialize:
        raise ValueError("Clean comparisons start from scratch")
    return target_epoch


def load_training_data(args):
    """Check saved split IDs/dimensions; select the PLD or DLMD pair loader."""
    identity = audit_manifest(args.manifest, args.dataset, smoke=args.smoke)
    if not {"train", "val"} <= identity["splits"].keys():
        raise ValueError("Training requires train and validation splits")
    training = dataset_for(args.dataset, args.manifest, args.pairs, "train")
    validation = dataset_for(args.dataset, args.manifest, args.pairs, "val")
    check_shape(training, args.dataset, smoke=args.smoke)
    check_shape(validation, args.dataset, smoke=args.smoke)
    return training, validation, identity


def make_training_loader(dataset, run, epoch, workers, data_rng):
    """Keep the recorded image order; ensemble members have separate data seeds.

    Paper runs draw a fresh permutation from data_seed + epoch. Baseline runs
    advance one persistent generator, whose state is saved/restored on resume.
    These policies are deliberately retained to reproduce the recorded runs.
    """
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
            prediction = center(model, method, measurement).float().clamp(0, 1)
        for name, values in reconstruction_metrics(prediction, target).items():
            totals[name] = totals.get(name, 0.0) + float(values.sum())
        image_count += len(batch)
    means = {name: total / image_count for name, total in totals.items()}
    return dict(count=image_count, clipped=True, **means)


def main():
    run_training(arguments())


if __name__ == "__main__":
    main()
