"""Calibrate frozen public models, then measure a separate held-out test split.

Baselines use RCPS-HB; QUTCC selects quantile inputs with paper Algorithm 1.
"""

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from convrml.calibration import CalibrationFailure, fit_calibration, apply_calibration
from convrml.checkpoints import load_predictor
from convrml.metrics import PerceptualMetric, finite_mean, interval_metrics
from convrml.protocol import (
    audit_manifest,
    autocast,
    dataset_for,
    environment,
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
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="One checkpoint, or ensemble members 0..4 / 0..9 in order",
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Manifest containing test IDs"
    )
    parser.add_argument(
        "--pairs", type=Path, help="PLD cache containing test pairs; omit for DLMD"
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        help="JSON containing disjoint calibration IDs; required for paper mode",
    )
    parser.add_argument(
        "--calibration-pairs",
        type=Path,
        help="Separate PLD calibration cache; omit for DLMD",
    )
    parser.add_argument(
        "--calibration",
        choices=["paper", "native"],
        default="paper",
        help="Paper: baseline RCPS or QUTCC Algorithm 1; native: uncalibrated control",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New result folder; existing calibration/report files are protected",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Explicitly omit pretrained LPIPS; recorded as unavailable, never zero",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Default: captured MC initialization seed; 20261022 for deterministic methods",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Save the first N test predictions and bounds as examples.npz",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Permit explicitly labelled small fixture data/checkpoints",
    )
    args = parser.parse_args(argv)
    if args.experiment is None:
        args.experiment = settings()["datasets"][args.dataset]["experiment"]
    return args


def split_ids(path, split):
    value = json.loads(Path(path).read_text())
    return [str(row["id"]) for row in value.get("roles", value)[split]]


def make_predictor(model, run, device):
    """Build a measurement (N,3,H,W) -> (center, lower, upper) callable.

    Inputs are in [0,1]; each output keeps the input shape but is unbounded.
    All methods share this interface despite their different forward() outputs.
    """

    def predict(measurement):
        with autocast(run, device):
            if run["method"] == "mc_dropout":
                result = model.bounds(
                    measurement,
                    samples=settings()["mc_samples"],
                    alpha=settings()["alpha"],
                )
            else:
                result = model.bounds(measurement)
        if not all(torch.isfinite(value).all() for value in result):
            raise ValueError("Nonfinite native predictions")
        return tuple(value.float() for value in result)

    return predict


def calibrate_checkpoint(
    args, run, model, predict, device, output, request, provenance
):
    """Fit and save calibration before loading test pixels.

    Returns (fitted parameters, calibration data identity, elapsed seconds).
    Native mode saves an empty fit; infeasible paper calibration writes failure.json.
    """
    started = time.monotonic()
    fitted, identity = {}, None
    if args.calibration != "native":
        identity = audit_manifest(
            args.calibration_manifest, args.dataset, smoke=args.smoke
        )
        calibration_ids = set(split_ids(args.calibration_manifest, "calibration"))
        manifest = json.loads(args.manifest.read_text())
        roles = manifest.get("roles", manifest)
        for role in ("train", "val", "test"):
            if role in roles and calibration_ids.intersection(
                split_ids(args.manifest, role)
            ):
                raise ValueError("Calibration IDs overlap another split")
        calibration = dataset_for(
            args.dataset,
            args.calibration_manifest,
            args.calibration_pairs,
            "calibration",
        )
        check_shape(calibration, args.dataset, smoke=args.smoke)
        try:
            fitted = fit_calibration(
                calibration, predict, model, run, device, args.workers, output
            )
        except CalibrationFailure as error:
            write_json(
                output / "failure.json",
                dict(
                    status="calibration_infeasible",
                    error_code="paper_calibration_infeasible",
                    error=str(error),
                    request=request,
                    provenance=provenance,
                ),
            )
            raise
    write_json(
        output / "calibration.json", dict(fitted, request=request, data=identity)
    )
    return fitted, identity, time.monotonic() - started


def summarize_metrics(rows):
    """Average each metric over defined per-image values and retain valid counts."""
    means, counts = {}, {}
    for space in ("raw", "clipped"):
        means[space], counts[space] = {}, {}
        for name in rows[0][space]:
            mean, count = finite_mean(row[space][name] for row in rows)
            means[space][name] = mean
            counts[space][name] = count
    return means, counts


def evaluate_test_set(
    dataset,
    ids,
    predict,
    *,
    batch_size,
    device,
    workers,
    example_count=0,
    perceptual=None,
):
    """Predict test images, measure metrics, and retain the first requested examples.

    ``ids`` follows dataset order; ``predict`` returns calibrated center/lower/upper.
    The result contains per-image rows, metric means/valid counts, timings and
    example arrays. Predictions/bounds in examples are clipped for display;
    reported metrics retain both raw and clipped views.
    """
    rows, offset = [], 0
    examples = {
        name: [] for name in ("measurement", "target", "prediction", "lower", "upper")
    }
    started = time.monotonic()
    prediction_seconds, metric_seconds = 0.0, 0.0
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
        measurement, target = batch[:, 0].to(device), batch[:, 1].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_started = time.monotonic()
        prediction, lower, upper = predict(measurement)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_seconds += time.monotonic() - prediction_started
        if not all(torch.isfinite(value).all() for value in (prediction, lower, upper)):
            raise ValueError("Nonfinite calibrated predictions")

        metric_started = time.monotonic()
        measured = {}
        for space in ("raw", "clipped"):
            measured[space] = interval_metrics(
                prediction, lower, upper, target, clipped=space == "clipped"
            )
            for name, values in measured[space].items():
                if (
                    name not in ("pearson", "spearman")
                    and not torch.isfinite(values).all()
                ):
                    raise ValueError("Nonfinite " + name + " scores")
        lpips = perceptual(prediction, target) if perceptual is not None else None
        if lpips is not None and not torch.isfinite(lpips).all():
            raise ValueError("Nonfinite LPIPS scores")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metric_seconds += time.monotonic() - metric_started

        for index in range(len(batch)):
            row = dict(index=offset + index, id=ids[offset + index])
            for space, values in measured.items():
                row[space] = {}
                for name, batch_values in values.items():
                    value = float(batch_values[index])
                    row[space][name] = value if math.isfinite(value) else None
            # LPIPS is defined on clipped images, not unbounded raw predictions.
            row["raw"]["lpips"] = None
            row["clipped"]["lpips"] = float(lpips[index]) if lpips is not None else None
            rows.append(row)
            if offset + index < example_count:
                images = dict(
                    measurement=measurement,
                    target=target,
                    prediction=prediction.clamp(0, 1),
                    lower=lower.clamp(0, 1),
                    upper=upper.clamp(0, 1),
                )
                for name, values in images.items():
                    examples[name].append(values[index].cpu().numpy())
        offset += len(batch)
    metrics, valid_counts = summarize_metrics(rows)
    timing = dict(
        test_seconds=time.monotonic() - started,
        prediction_seconds=prediction_seconds,
        prediction_seconds_per_image=prediction_seconds / len(rows),
        metric_seconds=metric_seconds,
    )
    return dict(
        rows=rows,
        examples=examples,
        metrics=metrics,
        valid_counts=valid_counts,
        timing=timing,
    )


def save_results(output, report, examples):
    """Write the aggregate report, per-image CSV and optional example arrays."""
    write_json(output / "report.json", report)
    columns = ["id", "index"]
    for space, metrics in report["metrics"].items():
        columns.extend(f"{space}_{name}" for name in metrics)
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in report["per_image"]:
            flattened = dict(id=row["id"], index=row["index"])
            for space in ("raw", "clipped"):
                for name, value in row[space].items():
                    flattened[f"{space}_{name}"] = value
            writer.writerow(flattened)
    if report["example_count"]:
        np.savez_compressed(
            output / "examples.npz",
            **{name: np.stack(values) for name, values in examples.items()},
        )


def run_evaluation(args, *, model_loader=None):
    """Evaluate args from arguments(), write result files, and return the report dict.

    A directory lock prevents concurrent writers. ``model_loader`` is an optional
    test hook; normal runs strictly load the checkpoint's recorded model identity.
    """
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_evaluation(args, model_loader=model_loader)


@torch.inference_mode()
def _run_evaluation(args, *, model_loader=None):
    if args.calibration not in ("paper", "native"):
        raise ValueError("Calibration must be paper or native")
    source, software = source_fingerprint(), environment()
    expected = (5, 10) if args.method == "ensemble" else (1,)
    if len(args.checkpoints) not in expected or args.workers < 0 or args.examples < 0:
        raise ValueError(
            "Expected one checkpoint, or five/ten ordered ensemble members"
        )
    data_identity = audit_manifest(args.manifest, args.dataset, smoke=args.smoke)
    test_ids = split_ids(args.manifest, "test")
    if args.examples > len(test_ids):
        raise ValueError("Requested examples exceed the held-out test set")
    if args.calibration != "native" and args.calibration_manifest is None:
        raise ValueError("Calibrated evaluation requires --calibration-manifest")
    device = torch.device(args.device)
    run = profile(
        args.experiment,
        args.dataset,
        args.method,
        0 if args.method == "ensemble" else None,
    )
    output = args.output.resolve()
    if any(
        (output / name).exists()
        for name in ("report.json", "calibration.json", "failure.json")
    ):
        raise FileExistsError(
            "Use a new output folder; existing fitted results are immutable"
        )

    # 1. Load frozen weights, then establish the inference random stream.
    started = time.monotonic()
    model, provenance = load_predictor(
        args.checkpoints, run, device, smoke=args.smoke, model_loader=model_loader
    )
    seed = args.seed
    if seed is None:
        seed = (
            run["seeds"]["initialization"] if args.method == "mc_dropout" else 20261022
        )
    torch.manual_seed(seed)
    load_seconds = time.monotonic() - started
    started = time.monotonic()
    perceptual = None if args.no_lpips else PerceptualMetric(device)
    lpips_load_seconds = time.monotonic() - started
    predict = make_predictor(model, run, device)
    request = dict(
        experiment=args.experiment,
        dataset=args.dataset,
        method=args.method,
        alpha=settings()["alpha"],
        calibration=args.calibration,
        seed=seed,
        mc_samples=settings()["mc_samples"] if args.method == "mc_dropout" else None,
        ensemble_members=len(provenance) if args.method == "ensemble" else None,
        device=str(device),
        smoke=args.smoke,
        test_used_for_calibration=False,
        lpips=not args.no_lpips,
    )

    # 2. Fit and save calibration before instantiating the test pixel loader.
    fitted, calibration_identity, calibration_seconds = calibrate_checkpoint(
        args,
        run,
        model,
        predict,
        device,
        output,
        request,
        provenance,
    )
    test = dataset_for(args.dataset, args.manifest, args.pairs, "test")
    check_shape(test, args.dataset, smoke=args.smoke)

    def calibrated_predict(measurement):
        return apply_calibration(measurement, predict, model, args.method, fitted)

    # 3. Predict held-out images and compute per-image metrics.
    result = evaluate_test_set(
        test,
        test_ids,
        calibrated_predict,
        batch_size=run["microbatch"],
        device=device,
        workers=args.workers,
        example_count=args.examples,
        perceptual=perceptual,
    )
    report = dict(
        epoch=provenance[0]["epoch"],
        camera=args.dataset,
        method=args.method,
        request=request,
        test_count=len(result["rows"]),
        metrics=result["metrics"],
        metric_valid_counts=result["valid_counts"],
        metric_protocol=dict(
            lpips=dict(
                enabled=perceptual is not None,
                backbone="alex",
                inputs="clipped RGB [0,1]",
                raw="unavailable: unbounded outputs outside supported intensity range",
            ),
            correlations="Per-image Pearson/Spearman: UB-LB versus absolute target-prediction error; flattened RGB; no smoothing/regions",
            undefined="null; means exclude undefined images and record valid counts",
        ),
        calibration=fitted,
        provenance=provenance,
        data=dict(main=data_identity, calibration=calibration_identity),
        environment=software,
        source_sha256=source,
        hardware=hardware(device),
        timing=dict(
            load_seconds=load_seconds,
            lpips_load_seconds=lpips_load_seconds,
            calibration_seconds=calibration_seconds,
            **result["timing"],
        ),
        per_image=result["rows"],
        example_count=args.examples,
    )
    # 4. Export the report and optional reconstructions/bounds.
    save_results(output, report, result["examples"])
    print(
        json.dumps(
            dict(test_count=report["test_count"], **report["metrics"]["clipped"])
        ),
        flush=True,
    )
    return report


def main():
    run_evaluation(arguments())


if __name__ == "__main__":
    main()
