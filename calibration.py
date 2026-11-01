"""Calibrate frozen public models, then measure a separate held-out test split.

Baselines use RCPS-HB; QUTCC selects quantile inputs with paper Algorithm 1.
Run ``python calibration.py --help`` for the model, data, and runtime options.
Calibration formulas live in convrml/calibration.py; held-out metrics and exports
live in convrml/evaluation.py. No model weights are updated by this workflow.
"""

import argparse
import fcntl
import json
from pathlib import Path
import time

import torch

from convrml.calibration import CalibrationFailure, fit_calibration, apply_calibration
from convrml.checkpoints import load_predictor
from convrml.metrics import PerceptualMetric
from convrml.evaluation import evaluate_test_set, save_results
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
    model_options = parser.add_argument_group("Model and checkpoints")
    data_options = parser.add_argument_group("Data splits")
    calibration_options = parser.add_argument_group("Calibration procedure")
    runtime_options = parser.add_argument_group("Runtime and outputs")
    model_options.add_argument(
        "--experiment", type=int, choices=[2, 3], help=argparse.SUPPRESS
    )
    data_options.add_argument(
        "--dataset", choices=list(settings()["datasets"]), required=True
    )
    model_options.add_argument(
        "--method", choices=["im2im", "ensemble", "mc_dropout", "qutcc"], required=True
    )
    model_options.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="One checkpoint, or ensemble members 0..4 / 0..9 in order",
    )
    data_options.add_argument(
        "--manifest", type=Path, required=True, help="Manifest containing test IDs"
    )
    data_options.add_argument(
        "--pairs", type=Path, help="PLD cache containing test pairs; omit for DLMD"
    )
    data_options.add_argument(
        "--calibration-manifest",
        type=Path,
        help="JSON containing disjoint calibration IDs; required for paper mode",
    )
    data_options.add_argument(
        "--calibration-pairs",
        type=Path,
        help="Separate PLD calibration cache; omit for DLMD",
    )
    calibration_options.add_argument(
        "--calibration",
        choices=["paper", "native"],
        default="paper",
        help="Paper: baseline RCPS or QUTCC Algorithm 1; native: uncalibrated control",
    )
    runtime_options.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New result folder; existing calibration/report files are protected",
    )
    runtime_options.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    runtime_options.add_argument("--workers", type=int, default=2)
    runtime_options.add_argument(
        "--no-lpips",
        action="store_true",
        help="Explicitly omit pretrained LPIPS; recorded as unavailable, never zero",
    )
    runtime_options.add_argument(
        "--seed",
        type=int,
        help="Default: captured MC initialization seed; 20261022 for deterministic methods",
    )
    runtime_options.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Save the first N test predictions and bounds as examples.npz",
    )
    runtime_options.add_argument(
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


def validate_request(args):
    """Reject invalid options and existing fitted outputs before loading weights."""
    if args.calibration not in ("paper", "native"):
        raise ValueError("Calibration must be paper or native")
    expected_checkpoints = (5, 10) if args.method == "ensemble" else (1,)
    if len(args.checkpoints) not in expected_checkpoints:
        raise ValueError(
            "Expected one checkpoint, or five/ten ordered ensemble members"
        )
    if args.workers < 0:
        raise ValueError("Worker count must be nonnegative")
    if args.examples < 0:
        raise ValueError("Example count must be nonnegative")
    if args.calibration != "native" and args.calibration_manifest is None:
        raise ValueError("Calibrated evaluation requires --calibration-manifest")
    protected_files = ("report.json", "calibration.json", "failure.json")
    if any((args.output / name).exists() for name in protected_files):
        raise FileExistsError(
            "Use a new output folder; existing fitted results are immutable"
        )


def load_frozen_checkpoint(args, run, device, *, model_loader=None):
    """Load verified weights, then seed the inference stream (including MC masks)."""
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
    return model, provenance, seed, load_seconds


def make_request(args, run, device, provenance, seed):
    """Record the requested method, calibration mode, and inference settings."""
    return dict(
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


def make_report(
    args,
    result,
    *,
    request,
    fitted,
    provenance,
    data,
    software,
    source,
    device,
    timing,
    lpips_enabled,
):
    """Assemble the exported metadata and measured scores without recomputing them."""
    return dict(
        epoch=provenance[0]["epoch"],
        camera=args.dataset,
        method=args.method,
        request=request,
        test_count=len(result["rows"]),
        metrics=result["metrics"],
        metric_valid_counts=result["valid_counts"],
        metric_protocol=dict(
            lpips=dict(
                enabled=lpips_enabled,
                backbone="alex",
                inputs="clipped RGB [0,1]",
                raw="unavailable: unbounded outputs outside supported intensity range",
            ),
            correlations="Per-image Pearson/Spearman: UB-LB versus absolute target-prediction error; flattened RGB; no smoothing/regions",
            undefined="null; means exclude undefined images and record valid counts",
        ),
        calibration=fitted,
        provenance=provenance,
        data=data,
        environment=software,
        source_sha256=source,
        hardware=hardware(device),
        timing={**timing, **result["timing"]},
        per_image=result["rows"],
        example_count=args.examples,
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
    validate_request(args)
    source, software = source_fingerprint(), environment()
    data_identity = audit_manifest(args.manifest, args.dataset, smoke=args.smoke)
    test_ids = split_ids(args.manifest, "test")
    if args.examples > len(test_ids):
        raise ValueError("Requested examples exceed the held-out test set")
    device = torch.device(args.device)
    run = profile(
        args.experiment,
        args.dataset,
        args.method,
        0 if args.method == "ensemble" else None,
    )
    output = args.output.resolve()

    # 1. Load frozen weights and initialize prediction services.
    model, provenance, seed, load_seconds = load_frozen_checkpoint(
        args, run, device, model_loader=model_loader
    )
    started = time.monotonic()
    perceptual = None if args.no_lpips else PerceptualMetric(device)
    lpips_load_seconds = time.monotonic() - started
    predict = make_predictor(model, run, device)
    request = make_request(args, run, device, provenance, seed)

    # 2. Fit and save calibration before opening any held-out test pixels.
    fitted, calibration_identity, calibration_seconds = calibrate_checkpoint(
        args,
        run=run,
        model=model,
        predict=predict,
        device=device,
        output=output,
        request=request,
        provenance=provenance,
    )
    test = dataset_for(args.dataset, args.manifest, args.pairs, "test")
    check_shape(test, args.dataset, smoke=args.smoke)

    def calibrated_predict(measurement):
        return apply_calibration(measurement, predict, model, args.method, fitted)

    # 3. Evaluate test images with the fixed calibration parameters.
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

    # 4. Assemble provenance and export scores, CSV rows, and optional examples.
    report = make_report(
        args,
        result,
        request=request,
        fitted=fitted,
        provenance=provenance,
        data=dict(main=data_identity, calibration=calibration_identity),
        software=software,
        source=source,
        device=device,
        timing=dict(
            load_seconds=load_seconds,
            lpips_load_seconds=lpips_load_seconds,
            calibration_seconds=calibration_seconds,
        ),
        lpips_enabled=perceptual is not None,
    )
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
