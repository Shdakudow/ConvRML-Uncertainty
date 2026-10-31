"""Fine-tune clean QUTCC/Im2im weights on online measurement modifications.

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

from convrml.checkpoints import load_model_weights, read_checkpoint, save_checkpoint
from dataset import ImagePairs
from input_modifications.online import CONDITIONS, _apply, sample_training_modification
from models import Im2im, QUTCC

STOP = 0
SETTINGS = json.loads(Path(__file__).with_name("protocol.json").read_text())


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
    ):
        paths.extend((REPOSITORY / name).rglob("*.py"))
    paths.append(Path(__file__).with_name("protocol.json"))
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


def train_batch(
    model, optimizer, batch, augmentation_rng, quantile_rng, *, check_gradients=False
):
    """Modify CPU measurements only, then accumulate an effective batch of four."""
    modified, labels = sample_training_modification(
        batch[:, 0], generator=augmentation_rng
    )
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
    model = QUTCC() if args.method == "qutcc" else Im2im()
    parent, parent_hash = read_checkpoint(args.initialize)
    config = parent["config"]
    if (
        parent_hash != SETTINGS["parent_sha256"][args.method]
        or parent["epoch"] != SETTINGS["parent_epoch"]
        or config["method"] != args.method
        or config["camera"] != "raw"
        or config["protocol"] != "paper_qutcc_convnext_uniform_pinball_20261018_v1"
        or SETTINGS["manifest_sha256"] not in config["dataset_identity"].values()
    ):
        raise ValueError(
            "Initialization is not the verified clean epoch-50 checkpoint."
        )
    load_model_weights(model, parent["model"])
    del parent
    model = model.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SETTINGS["learning_rate"], betas=tuple(SETTINGS["betas"])
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
        dataset="raw_online_ten_condition_mixture",
        initialization="fine_tune_verified_clean_epoch50_fresh_Adam_moments",
        parent_checkpoint_sha256=parent_hash,
        parent_path=str(args.initialize),
        manifest_path=str(args.manifest),
        source_sha256=source_hash(),
        source_commit=(REPOSITORY / "source_commit.txt").read_text().strip(),
        conditions=list(CONDITIONS),
        conditioning="blind; method labels never enter model",
        schedule="constant 1e-5 for 30 additional epochs",
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
            estimated_30_epoch_hours=30 * estimated_epoch_seconds / 3600,
            gpu=torch.cuda.get_device_name(),
            peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
        )
        write_json(args.output / "smoke.json", smoke)
        print(json.dumps(smoke), flush=True)
        return

    smoke = json.loads((args.output / "smoke.json").read_text())
    if not smoke["passed"] or smoke["config"]["source_sha256"] != meta["source_sha256"]:
        raise ValueError("A passing smoke run with identical source is required.")
    latest = args.output / "latest.pt"
    epoch, offset, global_step, loss_sum, loss_count = 1, 0, 0, 0.0, 0
    history, condition_counts = [], Counter()
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["config"] != meta:
            raise ValueError("Resume configuration changed.")
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
    if not latest.exists():
        baseline = validate(model, panel)
        write_json(args.output / "baseline_validation.json", baseline)
        checkpoint()
    if (
        offset == 0
        and epoch > 1
        and (epoch - 1) % SETTINGS["milestone_every_epochs"] == 0
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
        if completed_epoch % SETTINGS["milestone_every_epochs"] == 0:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qutcc", "im2im"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
