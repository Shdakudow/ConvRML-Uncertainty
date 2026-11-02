"""Calibrate saved models, then evaluate them on a separate held-out test split.

Reading order: main -> run_evaluation -> _run_evaluation. The workflow loads
frozen weights, fits interval parameters on calibration pairs, evaluates test
pairs with those parameters fixed, and saves the results. It never trains weights.

Paper mode uses separate lower/upper RCPS-HB scales for Im2im, ensemble and
MC dropout. QUTCC instead searches for two scalar q inputs shared by all images
using paper Algorithm 1. Native mode skips fitting and uses the original bounds.
Neither procedure forces exact test coverage or clips predictions to [0,1].
QUTCC's search assumes monotone quantile responses; it does not enforce them.

Outputs in --output:
    calibration.json: fitted scales or q values, request settings and split identity.
    report.json: aggregate/per-image raw and clipped metrics, provenance and timings.
    per_image.csv: one metric row per test image.
    examples.npz: optional first N measurements/targets and clipped predictions/bounds.
    failure.json: an explicit record when paper calibration is infeasible.

The fitting formulas are in convrml/calibration.py; metrics and exports are in
convrml/evaluation.py. Run ``python calibration.py --help`` for command options.
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
from convrml.evaluation import evaluate_test_set, make_report, save_results
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
    """Parse CLI options; an explicit argv list is useful for scripts and tests.

    --manifest supplies test IDs. Paper mode also needs calibration IDs, either
    in a separate manifest or a disjoint role in the same DLMD manifest. PLD uses
    NumPy pair caches; DLMD reads image paths from the manifest without --pairs.
    The dataset selects the recorded experiment profile automatically.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    model_options = parser.add_argument_group("Model and checkpoints")
    data_options = parser.add_argument_group("Data splits")
    calibration_options = parser.add_argument_group("Calibration procedure")
    runtime_options = parser.add_argument_group("Runtime and outputs")
    model_options.add_argument(
        # Retained for older commands; new commands only need --dataset.
        "--experiment",
        type=int,
        choices=[2, 3],
        help=argparse.SUPPRESS,
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


def manifest_ids(path):
    """Return {split: ordered image-ID strings}, without reading image pixels.

    Accept both PLD's top-level split keys and DLMD's nested ``roles`` mapping.
    Order matters: test IDs must align with loader order in the exported CSV.
    """
    value = json.loads(Path(path).read_text())
    roles = value.get("roles", value)
    return {
        role: [str(row["id"]) for row in roles[role]]
        for role in ("train", "val", "test", "calibration")
        if role in roles
    }


def load_calibration_split(args):
    """Return (calibration dataset, identity record) after checking split isolation.

    audit_manifest checks the saved split counts/order unless --smoke is set.
    The overlap check still runs in smoke mode: calibration IDs must not appear
    in train, validation or test. Identity records manifest and ordered-ID hashes.
    """
    identity = audit_manifest(args.calibration_manifest, args.dataset, smoke=args.smoke)
    calibration_ids = set(manifest_ids(args.calibration_manifest)["calibration"])
    for role, ids in manifest_ids(args.manifest).items():
        if role != "calibration" and calibration_ids.intersection(ids):
            raise ValueError("Calibration IDs overlap another split")
    calibration = dataset_for(
        args.dataset, args.calibration_manifest, args.calibration_pairs, "calibration"
    )
    check_shape(calibration, args.dataset, smoke=args.smoke)
    return calibration, identity


def make_predictor(model, run, device):
    """Build a measurement (N,3,H,W) -> (center, lower, upper) callable.

    Inputs are floating RGB intensities in [0,1]. Each output has the same shape
    and intensity units, but is unbounded. These are native, uncalibrated bounds.
    ``run`` is the recorded model/precision profile, not an optimizer or trainer.

    MC dropout aggregates repeated stochastic passes. Other models implement
    their native bounds directly. QUTCC's calibrated q inputs are applied later
    by apply_calibration, rather than changing this native predictor.
    """

    config = settings()

    def predict(measurement):
        with autocast(run, device):
            if run["method"] == "mc_dropout":
                result = model.bounds(
                    measurement,
                    samples=config["mc_samples"],
                    alpha=config["alpha"],
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
    ``fitted`` contains baseline scales or QUTCC q endpoints, plus fitting/audit
    metadata. Native mode returns an empty fit and no calibration identity.

    ``request`` records inference options; ``provenance`` identifies checkpoint
    files and epochs. Both help diagnose an infeasible fit in failure.json.
    Such a failure is raised immediately, so no test metrics are exported.
    """
    started = time.monotonic()
    fitted, identity = {}, None
    if args.calibration != "native":
        calibration, identity = load_calibration_split(args)
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
        # Persist the fit before the caller opens the held-out test dataset.
        output / "calibration.json",
        dict(fitted, request=request, data=identity),
    )
    return fitted, identity, time.monotonic() - started


def validate_request(args):
    """Reject invalid options and existing fitted outputs before loading weights.

    Ensemble checkpoints must be supplied in member order; the checkpoint loader
    separately verifies their identities, seeds and matching epochs. A fresh
    output directory prevents silently replacing an earlier fit or failure.
    """
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
    """Return (model, checkpoint provenance, inference seed, load time in seconds).

    Seed after model construction so initialization cannot consume the random
    stream intended for MC inference. This makes repeated evaluations replayable;
    it does not change the training seeds stored in the checkpoint.
    ``model_loader`` is only an optional fixture hook for correctness checks.
    """
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


def make_request(args, device, provenance, seed):
    """Build the inference-settings record stored in calibration and test reports.

    Alpha is the target miss fraction (0.10 means nominal 90% coverage), not a
    dropout probability. The RCPS confidence budget delta is a separate setting
    used by the fitting code. Neither is chosen using test ground truth.
    """
    config = settings()
    return dict(
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


def run_evaluation(args, *, model_loader=None):
    """Evaluate args from arguments(), write result files, and return the report dict.

    A directory lock prevents concurrent writers. ``model_loader`` is an optional
    test hook; normal runs strictly load the checkpoint's recorded model identity.
    The lock is released automatically on success or error. Existing report/fit
    files remain protected even after the lock is released.
    """
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_evaluation(args, model_loader=model_loader)


@torch.inference_mode()
def _run_evaluation(args, *, model_loader=None):
    """Run the four workflow stages under the output lock and inference_mode.

    inference_mode disables gradient recording. Loading/checkpoint utilities set
    model evaluation behavior; MC bounds deliberately enable their dropout masks.
    Test IDs can be inspected up front, but test pixels are opened only after the
    fit has been saved. Neither calibration nor test ground truth updates weights.
    """
    validate_request(args)
    # Record the exact source (including documentation) and software versions.
    source, software = source_fingerprint(), environment()
    data_identity = audit_manifest(args.manifest, args.dataset, smoke=args.smoke)
    test_ids = manifest_ids(args.manifest)["test"]
    if args.examples > len(test_ids):
        raise ValueError("Requested examples exceed the held-out test set")
    device = torch.device(args.device)
    run = profile(
        args.experiment,
        args.dataset,
        args.method,
        # Member zero supplies the common ensemble inference profile. The loader
        # validates every supplied member with its own initialization/data seeds.
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
    request = make_request(args, device, provenance, seed)

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
        # QUTCC queries the fitted q values; baselines scale native distances
        # from the center. In native mode fitted={} leaves predictions unchanged.
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
    # Raw/clipped views are reported separately; clipping is not calibration.
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
            environment=software,
            source_sha256=source,
            hardware=hardware(device),
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


def main():
    """CLI entry point; run_evaluation is the equivalent callable Python API."""
    run_evaluation(arguments())


if __name__ == "__main__":
    main()
