"""Calibrate frozen weights, then measure a separate test split.

Baselines use asymmetric RCPS-HB scales. QUTCC uses Algorithm 1 to select q
inputs. Calibration never updates weights or forces exact test coverage.
Saved outputs: calibration.json, report.json, per_image.csv and optional examples.npz.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from dataset import read_split_file

from convrml.calibration import CalibrationFailure, fit_calibration, apply_calibration
from convrml.checkpoints import load_predictor
from evaluation.reporting import evaluate_test_set, make_report, save_results
from evaluation.metrics import PerceptualMetric
from dataset import audit_manifest, dataset_for
from training import autocast, profile, settings
from convrml.checkpoints import write_json


def arguments(argv=None):
    """Read model, data and output choices; dataset selects its recorded settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=list(settings()["datasets"]), required=True
    )
    parser.add_argument(
        "--method", choices=["im2im", "ensemble", "mc_dropout", "qutcc"], required=True
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Test split JSON.")
    parser.add_argument("--pairs", type=Path, help="PLD NumPy cache; omit for DLMD.")
    parser.add_argument(
        "--calibration-manifest", type=Path, help="Separate calibration IDs."
    )
    parser.add_argument("--calibration-pairs", type=Path, help="PLD calibration cache.")
    parser.add_argument("--calibration", choices=["paper", "native"], default="paper")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, help="Inference RNG seed, including MC masks."
    )
    parser.add_argument(
        "--no-lpips", action="store_true", help="Skip pretrained LPIPS."
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Allow smoke checkpoint weights."
    )
    args = parser.parse_args(argv)
    args.experiment = settings()["datasets"][args.dataset]["experiment"]
    return args


@torch.inference_mode()
def run_evaluation(args, *, model_loader=None):
    """Load weights -> fit calibration -> predict test images -> save scores.

    Native mode skips calibration. Tests may supply a tiny model_loader;
    normal execution loads the selected model's weights strictly.
    """
    if args.calibration not in ("paper", "native"):
        raise ValueError("Calibration must be paper or native")
    if args.method != "ensemble" and len(args.checkpoints) != 1:
        raise ValueError("Expected one checkpoint, or ordered ensemble members")
    if args.workers < 0 or args.examples < 0:
        raise ValueError("Worker/example counts must be nonnegative")
    if args.calibration == "paper" and args.calibration_manifest is None:
        raise ValueError("Paper calibration requires --calibration-manifest")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in ("report.json", "calibration.json", "failure.json")
    ):
        raise FileExistsError("Use a new output folder for evaluation")
    data_identity = audit_manifest(args.manifest)
    if "test" not in data_identity["splits"]:
        raise ValueError("Evaluation requires a test split")
    if args.examples > data_identity["splits"]["test"]["count"]:
        raise ValueError("Requested examples exceed the test set")

    # 1. Load weights and seed inference after model construction.
    device = torch.device(args.device)
    run = profile(
        args.experiment,
        args.dataset,
        args.method,
        0 if args.method == "ensemble" else None,
    )
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
    config = settings()

    def predict(measurement):
        with autocast(run, device):
            if args.method == "mc_dropout":
                bounds = model.bounds(
                    measurement, samples=config["mc_samples"], alpha=config["alpha"]
                )
            else:
                bounds = model.bounds(measurement)
        if not all(torch.isfinite(value).all() for value in bounds):
            raise ValueError("Nonfinite native predictions")
        return tuple(value.float() for value in bounds)

    request = dict(
        experiment=args.experiment,
        dataset=args.dataset,
        method=args.method,
        alpha=config["alpha"],
        calibration=args.calibration,
        seed=seed,
        mc_samples=config["mc_samples"] if args.method == "mc_dropout" else None,
        ensemble_members=len(provenance) if args.method == "ensemble" else None,
        device=str(device),
        smoke=args.smoke,
        test_used_for_calibration=False,
        lpips=not args.no_lpips,
    )

    # 2. Fit using calibration targets only; reject overlap with any other role.
    started = time.monotonic()
    fitted, calibration_identity = {}, None
    if args.calibration == "paper":
        calibration_identity = audit_manifest(args.calibration_manifest)
        calibration = dataset_for(
            args.dataset,
            args.calibration_manifest,
            args.calibration_pairs,
            "calibration",
        )
        calibration_ids = set(calibration.ids)
        splits = read_split_file(args.manifest)
        for role in ("train", "val", "test"):
            if calibration_ids.intersection(
                str(row["id"]) for row in splits.get(role, [])
            ):
                raise ValueError("Calibration IDs overlap another split")
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
        output / "calibration.json",
        dict(fitted, request=request, data=calibration_identity),
    )
    calibration_seconds = time.monotonic() - started

    # 3. Open test pixels only after the fit has been saved.
    test = dataset_for(args.dataset, args.manifest, args.pairs, "test")

    def calibrated_predict(measurement):
        return apply_calibration(measurement, predict, model, args.method, fitted)

    result = evaluate_test_set(
        test,
        list(test.ids),
        calibrated_predict,
        batch_size=run["microbatch"],
        device=device,
        workers=args.workers,
        example_count=args.examples,
        perceptual=perceptual,
    )
    report = make_report(
        result,
        metadata=dict(
            epoch=provenance[0]["epoch"],
            camera=args.dataset,
            method=args.method,
            request=request,
            calibration=fitted,
            provenance=provenance,
            data=dict(main=data_identity, calibration=calibration_identity),
            timing=dict(
                load_seconds=load_seconds,
                lpips_load_seconds=lpips_load_seconds,
                calibration_seconds=calibration_seconds,
            ),
            example_count=args.examples,
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


if __name__ == "__main__":
    run_evaluation(arguments())
