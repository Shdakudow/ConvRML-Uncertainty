"""Readable cluster operations for the frozen-checkpoint evaluation pipeline.

The dispatcher sends JSON on stdin. This module restores verified checkpoints,
inspects results/jobs, submits once, and removes its own disposable checkpoint
copies. Model training and inference live elsewhere.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

RUN_ROOT = Path(__file__).resolve().parent
SLURM = Path("/usr/local/slurm/current/bin")
CAMERAS = ("rml", "diffuser")
BASELINE_ARMS = ("im2im", "mc_005", *(f"ensemble_{i:02d}" for i in range(5)))
LEGACY_CALIBRATION_ERRORS = {
    "No feasible extreme quantile endpoints",
    "Observed nonmonotone q response; calibration not certified",
    "Quantile endpoints reversed",
    "Final calibration recheck failed",
}


def write_json(path, value):
    """Replace a receipt/state file atomically, even if two clients reconnect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def checkpoint_paths(epoch):
    return {
        f"{camera}/{arm}/checkpoints/epoch_{epoch:03d}.pt"
        for camera in CAMERAS
        for arm in BASELINE_ARMS
    }


def validate_manifest(epoch, entries):
    paths = [entry["path"] for entry in entries]
    if (
        len(paths) != 14
        or len(set(paths)) != 14
        or set(paths) != checkpoint_paths(epoch)
    ):
        raise ValueError("Expected fourteen distinct camera/member checkpoints")


def task_folder(task):
    """Results must stay in this run's camera/method/epoch directory."""
    group, method, epoch = task["group"], task["method"], task["epoch"]
    allowed = (
        {"im2im", "qutcc"}
        if group in {"raw_clean", "raw_corrupted"}
        else {"im2im", "qutcc", "ensemble5", "mc_005"}
    )
    if (
        group not in {"rml", "diffuser", "raw_clean", "raw_corrupted"}
        or method not in allowed
    ):
        raise ValueError("Unknown experiment/method")
    if epoch not in range(5, (30 if group == "raw_corrupted" else 50) + 1, 5):
        raise ValueError("No such saved milestone in this experiment")
    folder = RUN_ROOT / "results" / group / method / f"epoch_{epoch:03d}"
    if "output" in task and Path(task["output"]).resolve() != folder.resolve():
        raise ValueError("Task output does not match its experiment/method/epoch")
    return folder


def validate_report(task, report, required_examples=0, check_aggregates=False):
    """A filename alone is not evidence that the requested model was evaluated."""
    request = report["request"]
    camera = task.get("camera") or (
        "raw" if task["group"].startswith("raw_") else task["group"]
    )
    if (request["camera"], request["method"], request["epoch"]) != (
        camera,
        task["method"],
        task["epoch"],
    ):
        raise ValueError("Saved report belongs to a different checkpoint")
    if (
        request["calibration"] != "tight_empirical"
        or request["test_used_for_calibration"] is not False
        or request["endpoint_clipping"] != [0, 1]
    ):
        raise ValueError("Saved report uses a different calibration protocol")
    if request["target_coverage"] != 0.9:
        raise ValueError("Saved report targets a different coverage")
    expected_test = 2000 if camera == "raw" else 5000
    if (
        report["test_count"] != expected_test
        or len(report["per_image"]) != expected_test
    ):
        raise ValueError("Incomplete held-out test evaluation")
    if len({str(row["id"]) for row in report["per_image"]}) != expected_test:
        raise ValueError("Duplicate test image IDs")
    if [row["index"] for row in report["per_image"]] != list(range(expected_test)):
        raise ValueError("Test image order is incomplete or inconsistent")
    calibration = report["calibration"]
    if (
        calibration["n_images"] != (3000 if camera == "raw" else 2048)
        or calibration["calibrated_on"] != "calibration split only"
        or calibration["confidence_guarantee"] is not False
    ):
        raise ValueError("Invalid calibration provenance")
    if calibration["target_coverage"] != 0.9 or calibration["endpoint_clipping"] != [
        0,
        1,
    ]:
        raise ValueError("Calibration and displayed metric conventions differ")
    members = (
        int(task["method"].removeprefix("ensemble"))
        if task["method"].startswith("ensemble")
        else 1
    )
    provenance = report["provenance"]
    if len(provenance) != members or any(
        row["epoch"] != task["epoch"] for row in provenance
    ):
        raise ValueError("Wrong ensemble size or mixed checkpoint epochs")
    if members > 1 and {row.get("arm") for row in provenance} != {
        f"ensemble_{i:02d}" for i in range(members)
    }:
        raise ValueError("Missing or duplicated ensemble members")
    if members > 1:
        for seed_name in ("initialization", "data"):
            if len({row["seeds"][seed_name] for row in provenance}) != members:
                raise ValueError(
                    "Ensemble members must have distinct initialization and data-order seeds"
                )
    if report["example_count"] < required_examples:
        raise ValueError(
            "Saved metrics lack the requested examples; use a separate export/output folder"
        )
    for mode in ("raw", "clipped"):
        for name in ("psnr", "ssim", "mae", "coverage", "width", "crossing"):
            value = report["metrics"][mode][name]
            if not math.isfinite(value):
                raise ValueError("Nonfinite saved metrics")
            if check_aggregates:
                mean = (
                    math.fsum(row[mode][name] for row in report["per_image"])
                    / expected_test
                )
                if not math.isclose(mean, value, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(
                        "Aggregate metrics do not match the full test images"
                    )


def calibration_failure(task, failure):
    """Accept a structured failure for this task, never a matching log phrase."""
    if failure.get("status") != "calibration_infeasible":
        return False
    if failure.get("method") != task["method"] or failure.get("epoch") != task["epoch"]:
        raise ValueError("Failure record belongs to a different checkpoint")
    if failure.get("test_used_for_calibration") is not False:
        raise ValueError("Failure record has invalid calibration provenance")
    if failure.get("group", task["group"]) != task["group"]:
        raise ValueError("Failure record belongs to a different experiment")
    expected_camera = task.get("camera") or (
        "raw" if task["group"].startswith("raw_") else task["group"]
    )
    if failure.get("camera", expected_camera) != expected_camera:
        raise ValueError("Failure record belongs to a different camera")
    if (
        failure.get("error_code")
        not in {
            "infeasible_bounds",
            "nonmonotone_response",
            "reversed_quantiles",
            "failed_recheck",
        }
        and failure.get("error") not in LEGACY_CALIBRATION_ERRORS
    ):
        raise ValueError("Unrecognized calibration failure")
    return True


def finished(task):
    folder = task_folder(task)
    report = folder / "report.json"
    if report.exists():
        validate_report(task, json.loads(report.read_text()))
        return True
    failure = folder / "failure.json"
    return failure.exists() and calibration_failure(
        task, json.loads(failure.read_text())
    )


def jobs():
    output = subprocess.check_output(
        [str(SLURM / "squeue"), "-u", "jy2228", "-h", "-o", "%F|%100j|%T"],
        text=True,
        timeout=60,
    )
    return [
        dict(zip(("job", "name", "state"), (part.strip() for part in line.split("|"))))
        for line in output.splitlines()
    ]


def status(request):
    job = str(request["job"])
    active = [row for row in jobs() if row["job"] == job]
    complete, missing = [], []
    for task in request["tasks"]:
        (complete if finished(task) else missing).append(str(task_folder(task)))
    return dict(active=active, complete=complete, missing=missing)


def valid_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == entry["sha256"]


def inventory(request):
    epoch, entries = request["epoch"], request["entries"]
    validate_manifest(epoch, entries)
    root = RUN_ROOT / "weights" / f"epoch_{epoch:03d}"
    root.mkdir(parents=True, exist_ok=True)
    shared = RUN_ROOT.parent / "experiment2_uncertainty_analysis_20261019/weights"
    missing = []
    for entry in entries:
        path, source = root / entry["path"], shared / entry["path"]
        if valid_file(path, entry):
            continue
        if not path.exists() and valid_file(source, entry):
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link(source.resolve(), path)
        else:
            missing.append(entry["path"])
    return dict(missing=missing)


def download_checkpoint(root, entry):
    """Download on Unicorn; publish a checkpoint only after its hash matches."""
    target = root / entry["path"]
    if valid_file(target, entry):
        return 0
    url = urllib.parse.urlsplit(entry["download_url"])
    if url.scheme != "https" or url.hostname != "release-assets.githubusercontent.com":
        raise ValueError("Expected a temporary GitHub Release download URL")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".pt.partial")
    transferred = 0
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= entry["size"]:
                if valid_file(partial, entry):
                    partial.replace(target)
                    return transferred
                partial.unlink()
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(entry["download_url"], headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                # An ignored Range request must replace the partial file.
                append = offset > 0 and response.status == 206
                written = offset if append else 0
                with partial.open("ab" if append else "wb") as stream:
                    while block := response.read(8 * 1024 * 1024):
                        written += len(block)
                        if written > entry["size"]:
                            raise ValueError("Downloaded checkpoint exceeds manifest size")
                        stream.write(block)
                        transferred += len(block)
            if not valid_file(partial, entry):
                partial.unlink(missing_ok=True)
                raise ValueError("Checkpoint SHA-256 or size does not match archive")
            partial.replace(target)
            return transferred
        except Exception:
            if attempt == 2:
                # Never log signed URLs from an HTTP exception.
                raise RuntimeError(
                    "Direct checkpoint download failed: " + entry["path"]
                ) from None
    raise RuntimeError("Checkpoint was not restored")


def restore(request):
    """Fetch four files at a time, directly from GitHub to cluster storage."""
    epoch, entries = request["epoch"], request["entries"]
    validate_manifest(epoch, entries)
    root = RUN_ROOT / "weights" / f"epoch_{epoch:03d}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".restore.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            sizes = list(
                pool.map(lambda entry: download_checkpoint(root, entry), entries)
            )
        remaining = inventory(request)["missing"]
        if remaining:
            raise ValueError("Incomplete direct checkpoint restoration")
        result = dict(
            epoch=epoch,
            verified=len(entries),
            downloaded_bytes=sum(sizes),
            seconds=round(time.monotonic() - started, 2),
            route="github-to-unicorn",
        )
        write_json(root / "restoration.json", result)
        return result


def submit(request):
    """Recover an accepted Slurm job after a lost SSH response instead of duplicating it."""
    tasks = request["tasks"]
    indices = sorted(task["index"] for task in tasks)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Empty or duplicate array indices")
    kind = request["kind"]
    if kind not in {"baseline", "paper-retry"}:
        raise ValueError("Unknown evaluation stream")
    parallelism = request.get("parallelism", 1)
    if type(parallelism) is not int or not 1 <= parallelism <= 6:
        raise ValueError("Evaluation parallelism must be an integer from one to six")
    if kind != "baseline" and parallelism != 1:
        raise ValueError("Only baseline evaluations use parallel GPU tasks")
    token = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()[:10]
    name = f"cov90-{kind}-{request['epoch']:03d}-{token}"
    directory = RUN_ROOT / "dispatch"
    directory.mkdir(exist_ok=True)
    receipt = directory / (name + ".json")
    # One lock for the entire stream also prevents two different epochs from
    # being submitted concurrently by reconnecting clients.
    with (directory / (kind + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = jobs()
        if receipt.exists():
            # The local dispatcher submits only missing tasks on a retry. A
            # finished submission of this same subset can legitimately retry.
            previous = json.loads(receipt.read_text())
            if any(row["job"] == previous["job"] for row in active) or all(
                finished(task) for task in tasks
            ):
                return previous
        if all(finished(task) for task in tasks):
            return dict(job=None, complete=True, recovered=True)
        existing = {row["job"] for row in active if row["name"] == name}
        if len(existing) > 1:
            raise RuntimeError("Duplicate active submissions require review")
        if existing:
            result = dict(job=existing.pop(), recovered=True)
        else:
            prefix = "cov90-baseline-" if kind == "baseline" else "cov90-paper-"
            if any(row["name"].startswith(prefix) for row in active):
                raise RuntimeError(
                    "Another array in this evaluation stream is still active"
                )
            command = [
                str(SLURM / "sbatch"),
                "--parsable",
                "--array=" + ",".join(map(str, indices)) + f"%{parallelism}",
                "--job-name=" + name,
            ]
            if kind == "baseline":
                command.append("--partition=gpu")
            command.append("milestone.sbatch")
            job = (
                subprocess.check_output(command, cwd=RUN_ROOT, text=True)
                .strip()
                .split(";")[0]
            )
            if not re.fullmatch(r"\d+", job):
                raise RuntimeError("Slurm did not return a valid job ID")
            result = dict(job=job, recovered=False, parallelism=parallelism)
        write_json(receipt, result)
        return result


def cleanup(request):
    epoch, entries, tasks = request["epoch"], request["entries"], request["tasks"]
    validate_manifest(epoch, entries)
    expected = {
        (camera, method)
        for camera in CAMERAS
        for method in ("im2im", "ensemble5", "mc_005")
    }
    actual = {(task["group"], task["method"]) for task in tasks}
    if (
        len(tasks) != 6
        or actual != expected
        or any(task["epoch"] != epoch for task in tasks)
    ):
        raise ValueError("Cleanup requires all six baseline evaluations at this epoch")
    if not tasks or not all(finished(task) for task in tasks):
        raise ValueError("Cannot remove checkpoints before their evaluations finish")
    root = RUN_ROOT / "weights" / f"epoch_{epoch:03d}"
    for entry in entries:
        (root / entry["path"]).unlink(missing_ok=True)
    return dict(cleaned_epoch=epoch)


def curves(_request):
    """Collect compact, validated metrics; never interpolate missing checkpoints."""
    methods = {
        "rml": ("im2im", "ensemble5", "mc_005", "qutcc"),
        "diffuser": ("im2im", "ensemble5", "mc_005", "qutcc"),
        "raw_clean": ("im2im", "qutcc"),
        "raw_corrupted": ("im2im", "qutcc"),
    }
    output = dict(
        protocol="tight_empirical_coverage90",
        target_calibration_coverage=[0.9, 0.901],
        groups={},
    )
    for group, names in methods.items():
        output["groups"][group] = {}
        last_epoch = 30 if group == "raw_corrupted" else 50
        for method in names:
            latest_task = dict(group=group, method=method, epoch=last_epoch)
            latest = json.loads((task_folder(latest_task) / "report.json").read_text())
            validate_report(latest_task, latest)
            rows = []
            for epoch in range(5, last_epoch + 1, 5):
                task = dict(group=group, method=method, epoch=epoch)
                folder = task_folder(task)
                row = dict(epoch=epoch, status="pending")
                report_path, failure_path = (
                    folder / "report.json",
                    folder / "failure.json",
                )
                if report_path.exists():
                    raw = report_path.read_bytes()
                    report = json.loads(raw)
                    validate_report(task, report, check_aggregates=True)
                    if report["split_identity"] != latest["split_identity"]:
                        raise ValueError(
                            "Milestone and latest checkpoint have different dataset splits"
                        )
                    row.update(
                        status="complete",
                        metrics=report["metrics"],
                        calibration=report["calibration"],
                        test_count=report["test_count"],
                        report_sha256=hashlib.sha256(raw).hexdigest(),
                    )
                elif failure_path.exists():
                    failure = json.loads(failure_path.read_text())
                    if calibration_failure(task, failure):
                        row.update(status="failed", error=failure["error"])
                rows.append(row)
            output["groups"][group][method] = rows
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = dict(
        inventory=inventory,
        restore=restore,
        status=status,
        submit=submit,
        cleanup=cleanup,
        curves=curves,
    )
    parser.add_argument("action", choices=actions)
    args = parser.parse_args()
    import sys

    print(json.dumps(actions[args.action](json.load(sys.stdin)), allow_nan=False))


if __name__ == "__main__":
    main()
