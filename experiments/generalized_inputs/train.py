"""Train QUTCC/Im2im from scratch on online measurement modifications.

Run from a frozen repository snapshot. No modified datasets, calibration fitting,
or test evaluation are created by this training entry point.
"""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from convrml.checkpoints import load_model_weights, save_checkpoint
from dataset import ImagePairs
from input_modifications.gaussian import gaussian_noise
from input_modifications.online import CONDITIONS, _apply, sample_training_modification
from models import Im2im, QUTCC

STOP = 0
SETTINGS_PATH = Path(__file__).with_name("protocol.json")
SETTINGS = json.loads(SETTINGS_PATH.read_text())


def request_stop(signum, _frame):
    global STOP
    STOP = 75 if signum == signal.SIGUSR1 else 1


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash():
    paths = [REPOSITORY / "dataset.py"]
    for name in (
        "convrml",
        "models",
        "input_modifications",
        "experiments/generalized_inputs",
        "experiments/gaussian_specialist",
    ):
        paths.extend((REPOSITORY / name).rglob("*.py"))
    paths.append(SETTINGS_PATH)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPOSITORY)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def audit_manifest(path):
    if file_hash(path) != SETTINGS["manifest_sha256"]:
        raise ValueError("The original MIRFLICKR split manifest changed.")
    roles = json.loads(path.read_text())["roles"]
    seen = set()
    for role, count in SETTINGS["counts"].items():
        ids = {str(row["id"]) for row in roles[role]}
        if len(roles[role]) != count or len(ids) != count or seen.intersection(ids):
            raise ValueError(f"Invalid count or overlapping IDs in {role}.")
        seen.update(ids)


def initialize_model(method):
    """Create fresh random weights; both backbones use pretrained=False."""
    torch.manual_seed(SETTINGS["seed"])
    return QUTCC() if method == "qutcc" else Im2im()


def weights_hash(model):
    """Record the complete initial state so smoke and full runs can be compared."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_initial_weights(
    model, output, method, smoke, previous_config=None, initial_reference=None
):
    """Save fresh epoch-zero weights before smoke; replay them across CPU types."""
    path = output / "initial_random.pt"
    expected = dict(
        epoch=0,
        trained_steps=0,
        pretrained=False,
        method=method,
        seed=SETTINGS["seed"],
        protocol=(previous_config or SETTINGS)["protocol"],
        source_sha256=(
            previous_config["source_sha256"] if previous_config else source_hash()
        ),
    )
    matched_hash = SETTINGS.get("matched_initial_weights_sha256", {}).get(method)
    if matched_hash:
        expected["matched_initial_weights_sha256"] = matched_hash
    if smoke:
        if path.exists():
            raise FileExistsError("Fresh initialization already exists; use a new run.")
        if matched_hash:
            if initial_reference is None:
                raise ValueError(
                    "The matched experiment requires an untrained epoch-zero reference."
                )
            reference = torch.load(
                initial_reference, map_location="cpu", weights_only=False
            )
            required = dict(
                epoch=0,
                trained_steps=0,
                pretrained=False,
                method=method,
                seed=SETTINGS["seed"],
            )
            if any(reference.get(key) != value for key, value in required.items()):
                raise ValueError(
                    "The initialization reference must have zero trained steps."
                )
            load_model_weights(model, reference["model"])
            del reference
        digest = weights_hash(model)
        if matched_hash and digest != matched_hash:
            raise ValueError(
                "The untrained reference does not match the comparison's initial weights."
            )
        save_checkpoint(
            path, dict(**expected, model=model.state_dict(), weights_sha256=digest)
        )
        return digest
    initial = torch.load(path, map_location="cpu", weights_only=False)
    if any(initial.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Initial state must be this run's untrained epoch-zero weights."
        )
    load_model_weights(model, initial["model"])
    digest = weights_hash(model)
    if digest != initial["weights_sha256"] or (matched_hash and digest != matched_hash):
        raise ValueError("Untrained epoch-zero weight hash mismatch.")
    return digest


def learning_rate(step, total_steps):
    """Keep the first 50 epochs intact, then use a separate lower-rate phase."""
    if "base_epochs" in SETTINGS:
        boundary = total_steps * SETTINGS["base_epochs"] / SETTINGS["epochs"]
        if step >= boundary:
            fraction = min((step - boundary) / (total_steps - boundary), 1.0)
            cosine = (1 + math.cos(math.pi * fraction)) / 2
            return (
                SETTINGS["continuation_final_lr"]
                + (
                    SETTINGS["continuation_initial_lr"]
                    - SETTINGS["continuation_final_lr"]
                )
                * cosine
            )
        total_steps = boundary
    fraction = min(step / total_steps, 1.0)
    warmup = SETTINGS["warmup_fraction"]
    if fraction <= warmup:
        return SETTINGS["initial_lr"] + (fraction / warmup) * (
            SETTINGS["peak_lr"] - SETTINGS["initial_lr"]
        )
    cosine = (1 + math.cos(math.pi * (fraction - warmup) / (1 - warmup))) / 2
    return SETTINGS["final_lr"] + (SETTINGS["peak_lr"] - SETTINGS["final_lr"]) * cosine


def validate_epoch_extension(previous, current):
    """Allow the reviewed epoch extensions while retaining training identity."""
    allowed = {
        "protocol",
        "epochs",
        "warmup_fraction",
        "schedule",
        "source_sha256",
        "source_commit",
        "epoch_extension",
    }
    if previous["epochs"] == 50 and current["epochs"] == 100:
        allowed.update(
            {"base_epochs", "continuation_initial_lr", "continuation_final_lr"}
        )
    changed = {
        key
        for key in previous.keys() | current.keys()
        if previous.get(key) != current.get(key)
    }
    if changed - allowed:
        raise ValueError(
            f"Epoch extension changed training identity: {changed - allowed}"
        )
    if previous["epochs"] == 50 and current["epochs"] == 100:
        if current["base_epochs"] != 50 or not (
            0
            < current["continuation_final_lr"]
            <= current["continuation_initial_lr"]
            < previous["peak_lr"]
        ):
            raise ValueError(
                "The continuation must start after 50 at a lower peak rate."
            )
        if previous["warmup_fraction"] != current["warmup_fraction"]:
            raise ValueError("The original 50-epoch schedule must remain unchanged.")
    elif previous["epochs"] != 30 or current["epochs"] != 50:
        raise ValueError(
            "Only the requested 30-to-50 or 50-to-100 extensions are allowed."
        )
    elif not math.isclose(
        previous["epochs"] * previous["warmup_fraction"],
        current["epochs"] * current["warmup_fraction"],
    ):
        raise ValueError("Epoch extension must retain the original warmup duration.")
    if (
        previous["initialization"] != "fresh_random_no_pretrained_weights"
        or previous["pretrained"]
    ):
        raise ValueError("Only the existing from-scratch runs can be extended.")


def validate_after50_checkpoint(state, complete):
    """Require every epoch-50 update and validation before phase two can start."""
    expected_steps = 50 * math.ceil(
        SETTINGS["counts"]["train"] / SETTINGS["effective_batch"]
    )
    if (
        not complete.get("passed")
        or complete.get("epochs") != 50
        or state["epoch"] != 50
        or state["next_epoch"] != 51
        or state["offset"] != 0
        or state["global_step"] != expected_steps
        or [row["epoch"] for row in state["history"]] != list(range(1, 51))
    ):
        raise ValueError("Epochs 51-100 require a complete validated 50-epoch run.")


class ValidationPanel(Dataset):
    """The same held-out scenes under every condition, with fixed random draws."""

    def __init__(self, original, count):
        self.original = original
        self.indices = torch.randperm(
            len(original),
            generator=torch.Generator().manual_seed(SETTINGS["validation_seed"]),
        )[:count].tolist()

    def __len__(self):
        return len(CONDITIONS) * len(self.indices)

    def __getitem__(self, index):
        condition_index, scene_index = divmod(index, len(self.indices))
        source_index = self.indices[scene_index]
        measurement, target = self.original[source_index]
        generator = torch.Generator().manual_seed(
            SETTINGS["validation_seed"] + 10000 * condition_index + source_index
        )
        modified = _apply(
            measurement.unsqueeze(0), CONDITIONS[condition_index], generator
        )[0]
        return modified, target, condition_index


@torch.inference_mode()
def validate(model, panel):
    model.eval()
    totals = {name: Counter() for name in CONDITIONS}
    loader = DataLoader(panel, batch_size=1, num_workers=2, pin_memory=True)
    for measurement, target, condition in loader:
        target = target.cuda(non_blocking=True)
        prediction = model.center(measurement.cuda(non_blocking=True)).clamp(0, 1)
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite validation prediction.")
        row = totals[CONDITIONS[condition.item()]]
        row["psnr"] += float(
            -10 * (prediction - target).square().mean().clamp_min(1e-12).log10()
        )
        row["ssim"] += float(ssim(prediction, target, data_range=1.0))
        row["mae"] += float((prediction - target).abs().mean())
        row["count"] += 1
    conditions = {
        name: dict(
            count=row["count"],
            **{key: row[key] / row["count"] for key in ("psnr", "ssim", "mae")},
        )
        for name, row in totals.items()
    }
    return dict(
        conditions=conditions,
        macro_psnr=sum(row["psnr"] for row in conditions.values()) / len(CONDITIONS),
        count=len(panel),
        clipped=True,
        interval_calibration_performed=False,
    )


def modify_training_measurements(measurement, generator):
    """Select the protocol's input distribution; targets never enter this helper."""
    condition = SETTINGS.get("training_condition", "mixture")
    if condition == "mixture":
        return sample_training_modification(measurement, generator=generator)
    if condition == "gaussian":
        modified = gaussian_noise(
            measurement, sigma=SETTINGS["gaussian_sigma"], generator=generator
        )
        return modified, ["gaussian"] * len(measurement)
    raise ValueError(f"Unsupported training condition: {condition}")


def train_batch(
    model, optimizer, batch, augmentation_rng, quantile_rng, *, check_gradients=False
):
    """Modify CPU measurements only, then accumulate an effective batch of four."""
    modified, labels = modify_training_measurements(batch[:, 0], augmentation_rng)
    optimizer.zero_grad(set_to_none=True)
    value = 0.0
    for index in range(len(batch)):
        measurement = modified[index : index + 1].cuda(non_blocking=True)
        target = batch[index : index + 1, 1].cuda(non_blocking=True)
        loss = (
            model.loss(measurement, target, generator=quantile_rng)
            if isinstance(model, QUTCC)
            else model.loss(measurement, target)
        ) / len(batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss.")
        loss.backward()
        value += float(loss.detach())
    if check_gradients and not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise ValueError("Nonfinite training gradients.")
    optimizer.step()
    return value, labels


def run(args):
    audit_manifest(args.manifest)
    torch.set_num_threads(2)
    torch.manual_seed(SETTINGS["seed"])
    torch.cuda.manual_seed_all(SETTINGS["seed"])
    model = initialize_model(args.method)
    args.output.mkdir(parents=True, exist_ok=True)
    previous_config = json.loads(args.extension.read_text()) if args.extension else None
    initial_config = (
        json.loads(args.initial_config.read_text())
        if args.initial_config
        else previous_config
    )
    if (
        previous_config
        and previous_config["epochs"] == 50
        and SETTINGS["epochs"] == 100
        and not args.after50
    ):
        raise ValueError(
            "The 100-epoch continuation must wait for the completed epoch-50 run."
        )
    if previous_config and (args.smoke or not (args.output / "latest.pt").exists()):
        raise ValueError(
            "An epoch extension requires the existing run's resume checkpoint."
        )
    initial_weights_sha256 = prepare_initial_weights(
        model,
        args.output,
        args.method,
        args.smoke,
        initial_config,
        initial_reference=args.initial_random_state,
    )
    model = model.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SETTINGS["initial_lr"], betas=tuple(SETTINGS["betas"])
    )
    augmentation_rng = torch.Generator().manual_seed(SETTINGS["seed"] + 1000)
    quantile_rng = torch.Generator(device="cuda").manual_seed(SETTINGS["seed"] + 2000)
    train = ImagePairs(args.manifest, "train")
    val = ImagePairs(args.manifest, "val")
    panel = ValidationPanel(
        val, 2 if args.smoke else SETTINGS["validation_per_condition"]
    )
    meta = dict(
        **SETTINGS,
        method=args.method,
        dataset=(
            "raw_online_gaussian_sigma001"
            if SETTINGS.get("training_condition") == "gaussian"
            else "raw_online_ten_condition_mixture"
        ),
        pretrained=False,
        initial_weights_sha256=initial_weights_sha256,
        pretrained_checkpoint=None,
        initial_state_file="initial_random.pt",
        initial_trained_steps=0,
        manifest_path=str(args.manifest),
        source_sha256=source_hash(),
        source_commit=(REPOSITORY / "source_commit.txt").read_text().strip(),
        conditions=list(CONDITIONS),
        conditioning="blind; method labels never enter model",
        loss=(
            "sampled uniform-q pinball"
            if args.method == "qutcc"
            else "center MSE + lower .05 pinball + upper .95 pinball"
        ),
        validation_indices=panel.indices,
        test_used=False,
        calibration_used=False,
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda),
    )
    if previous_config:
        meta["epoch_extension"] = dict(
            previous_epochs=previous_config["epochs"],
            previous_source_sha256=previous_config["source_sha256"],
            previous_source_commit=previous_config["source_commit"],
            previous_config_sha256=file_hash(args.extension),
        )
        validate_epoch_extension(previous_config, meta)
        if (
            initial_config["initial_weights_sha256"]
            != previous_config["initial_weights_sha256"]
        ):
            raise ValueError("Epoch-zero provenance changed across continuations.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        model.train()
        before = next(model.parameters()).detach().clone()
        times, losses, conditions = [], [], Counter()
        loader = DataLoader(
            Subset(train, range(32)),
            batch_size=SETTINGS["effective_batch"],
            num_workers=2,
        )
        for batch in loader:
            torch.cuda.synchronize()
            started = time.monotonic()
            value, labels = train_batch(
                model,
                optimizer,
                batch,
                augmentation_rng,
                quantile_rng,
                check_gradients=True,
            )
            torch.cuda.synchronize()
            times.append(time.monotonic() - started)
            losses.append(value)
            conditions.update(labels)
        assert not torch.equal(before, next(model.parameters()))
        started = time.monotonic()
        validation = validate(model, panel)
        validation_seconds = time.monotonic() - started
        seconds_per_step = sum(times[1:]) / len(times[1:])
        estimated_epoch_seconds = seconds_per_step * math.ceil(
            len(train) / SETTINGS["effective_batch"]
        ) + validation_seconds / len(panel) * (
            len(CONDITIONS) * SETTINGS["validation_per_condition"]
        )
        smoke = dict(
            config=meta,
            passed=True,
            losses=losses,
            condition_counts=dict(conditions),
            validation=validation,
            seconds_per_step=seconds_per_step,
            estimated_epoch_hours=estimated_epoch_seconds / 3600,
            estimated_total_epoch_hours=SETTINGS["epochs"]
            * estimated_epoch_seconds
            / 3600,
            gpu=torch.cuda.get_device_name(),
            peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
        )
        write_json(args.output / "smoke.json", smoke)
        print(json.dumps(smoke), flush=True)
        return

    smoke = json.loads((args.output / "smoke.json").read_text())
    verified_source = (
        initial_config["source_sha256"] if initial_config else meta["source_sha256"]
    )
    if not smoke["passed"] or smoke["config"]["source_sha256"] != verified_source:
        raise ValueError("A passing smoke run with identical source is required.")
    if smoke["config"]["initial_weights_sha256"] != initial_weights_sha256:
        raise ValueError("Full run must start with the same fresh weights as smoke.")
    latest = args.output / "latest.pt"
    epoch, offset, global_step, loss_sum, loss_count = 1, 0, 0, 0.0, 0
    history, condition_counts = [], Counter()
    extension_applied = False
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["config"] != meta:
            if not previous_config or state["config"] != previous_config:
                raise ValueError("Resume configuration changed.")
            validate_epoch_extension(state["config"], meta)
            original_warmup_steps = (
                previous_config["epochs"]
                * previous_config["warmup_fraction"]
                * math.ceil(len(train) / SETTINGS["effective_batch"])
            )
            if args.after50:
                complete = json.loads(
                    (args.output / "COMPLETE_50epochs.json").read_text()
                )
                validate_after50_checkpoint(state, complete)
            elif state["global_step"] >= original_warmup_steps:
                raise ValueError(
                    "This extension must happen during the unchanged warmup."
                )
            extension_applied = True
        load_model_weights(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        augmentation_rng.set_state(state["augmentation_rng"])
        quantile_rng.set_state(state["quantile_rng"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        epoch, offset, global_step, history, loss_sum, loss_count = [
            state[key]
            for key in (
                "next_epoch",
                "offset",
                "global_step",
                "history",
                "loss_sum",
                "loss_count",
            )
        ]
        condition_counts = Counter(state["condition_counts"])
        del state

    def checkpoint():
        save_checkpoint(
            latest,
            dict(
                config=meta,
                epoch=epoch - 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                augmentation_rng=augmentation_rng.get_state(),
                quantile_rng=quantile_rng.get_state(),
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                next_epoch=epoch,
                offset=offset,
                global_step=global_step,
                history=history,
                loss_sum=loss_sum,
                loss_count=loss_count,
                condition_counts=dict(condition_counts),
            ),
        )

    def milestone(completed_epoch):
        path = args.output / "checkpoints" / f"epoch_{completed_epoch:03d}.pt"
        if not path.exists():
            save_checkpoint(
                path,
                dict(
                    config=meta,
                    epoch=completed_epoch,
                    model=model.state_dict(),
                    history=history,
                ),
            )

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    write_json(args.output / "config.json", meta)
    if extension_applied:
        checkpoint()
        write_json(
            args.output / "extension_applied.json",
            dict(
                previous_epochs=previous_config["epochs"],
                epochs=SETTINGS["epochs"],
                next_epoch=epoch,
                offset=offset,
                global_step=global_step,
                initial_weights_sha256=initial_weights_sha256,
                source_sha256=meta["source_sha256"],
            ),
        )
    if not latest.exists():
        baseline = validate(model, panel)
        write_json(args.output / "baseline_validation.json", baseline)
        checkpoint()
    if (
        offset == 0
        and epoch > 1
        and (
            (epoch - 1) % SETTINGS["milestone_every_epochs"] == 0
            or epoch - 1 == SETTINGS["epochs"]
        )
    ):
        milestone(epoch - 1)
    segment_started = time.monotonic()
    while epoch <= SETTINGS["epochs"]:
        order = torch.randperm(
            len(train),
            generator=torch.Generator().manual_seed(SETTINGS["seed"] + epoch),
        ).tolist()
        loader = DataLoader(
            Subset(train, order[offset:]),
            batch_size=SETTINGS["effective_batch"],
            num_workers=2,
            pin_memory=True,
            generator=torch.Generator().manual_seed(SETTINGS["seed"] + epoch + 10000),
        )
        model.train()
        epoch_started = time.monotonic()
        for batch in loader:
            lr = learning_rate(
                global_step,
                SETTINGS["epochs"]
                * math.ceil(len(train) / SETTINGS["effective_batch"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            value, labels = train_batch(
                model, optimizer, batch, augmentation_rng, quantile_rng
            )
            offset += len(batch)
            global_step += 1
            loss_sum += value * len(batch)
            loss_count += len(batch)
            condition_counts.update(labels)
            if global_step == 1 or global_step % 25 == 0:
                progress = dict(
                    method=args.method,
                    stage="training",
                    epoch=epoch,
                    epochs=SETTINGS["epochs"],
                    processed=offset,
                    total=len(train),
                    global_step=global_step,
                    loss=value,
                    learning_rate=lr,
                    condition_counts=dict(condition_counts),
                    gpu=torch.cuda.get_device_name(),
                )
                write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            requeue = (
                time.monotonic() - segment_started > SETTINGS["segment_hours"] * 3600
            )
            if global_step % SETTINGS["checkpoint_every_steps"] == 0 or STOP or requeue:
                checkpoint()
            if STOP or requeue:
                raise SystemExit(STOP or 75)
        validation = validate(model, panel)
        row = dict(
            epoch=epoch,
            training_loss=loss_sum / loss_count,
            learning_rate=optimizer.param_groups[0]["lr"],
            condition_counts=dict(condition_counts),
            validation=validation,
            training_and_validation_seconds=time.monotonic() - epoch_started,
            gpu=torch.cuda.get_device_name(),
        )
        history.append(row)
        write_json(args.output / "history.json", history)
        print(json.dumps(row), flush=True)
        completed_epoch = epoch
        epoch += 1
        offset, loss_sum, loss_count = 0, 0.0, 0
        condition_counts = Counter()
        checkpoint()
        if (
            completed_epoch % SETTINGS["milestone_every_epochs"] == 0
            or completed_epoch == SETTINGS["epochs"]
        ):
            milestone(completed_epoch)
        if STOP:
            raise SystemExit(STOP)
    write_json(
        args.output / "COMPLETE.json",
        dict(
            passed=True,
            epochs=SETTINGS["epochs"],
            method=args.method,
            final_validation=history[-1]["validation"],
            training_only=True,
            calibration_performed=False,
            test_evaluated=False,
        ),
    )


def main():
    global SETTINGS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=SETTINGS_PATH,
        help="Protocol JSON inside the frozen repository snapshot",
    )
    parser.add_argument(
        "--initial-random-state",
        type=Path,
        help="Verified untrained epoch-zero reference for a matched comparison",
    )
    parser.add_argument("--method", choices=("qutcc", "im2im"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--extension",
        type=Path,
        help="Original config.json recording the predecessor run for an epoch extension",
    )
    parser.add_argument(
        "--initial-config",
        type=Path,
        help="Original epoch-zero provenance for a chained continuation",
    )
    parser.add_argument(
        "--after50",
        action="store_true",
        help="Require a finished 50-epoch predecessor before continuing",
    )
    args = parser.parse_args()
    SETTINGS_PATH = args.protocol.resolve()
    SETTINGS.clear()
    SETTINGS.update(json.loads(SETTINGS_PATH.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
