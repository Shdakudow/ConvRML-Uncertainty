"""Calibrate frozen public models, then measure a separate held-out test split.

Uses no archived trainers, cluster paths or test-driven parameter selection.
Baselines use RCPS-HB; QUTCC selects quantile inputs with paper Algorithm 1.
"""

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import CalibrationFailure, paper_quantiles, rcps_scales
from .checkpoints import load_model
from .metrics import PerceptualMetric, finite_mean, interval_metrics
from .protocol import (
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
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--calibration-pairs", type=Path)
    parser.add_argument(
        "--calibration",
        choices=["paper", "native"],
        default="paper",
        help="Paper: baseline RCPS or QUTCC Algorithm 1; native: uncalibrated control",
    )
    parser.add_argument("--output", type=Path, required=True)
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


def capture_rng(device):
    return torch.get_rng_state(), (
        torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    )


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def fit_calibration(calibration, predict, q_model, run, device, workers, output):
    loader = DataLoader(calibration, batch_size=run["microbatch"], num_workers=workers)
    target = settings()["alpha"]
    rng_before, rng_after = capture_rng(device), None
    if run["method"] == "qutcc":
        cache = [{}, {}]

        def risk(qlo, qhi):
            qs = qlo, qhi
            missing = [side for side in (0, 1) if qs[side] not in cache[side]]
            counts, pixels = [0, 0], 0
            if missing:
                for batch in loader:
                    y, x = batch[:, 0].to(device), batch[:, 1].to(device)
                    for side in missing:
                        endpoint = q_model(y, y.new_full((len(y),), qs[side]))
                        if not torch.isfinite(endpoint).all():
                            raise ValueError("Nonfinite quantile prediction")
                        counts[side] += int(
                            ((x < endpoint) if side == 0 else (x > endpoint)).sum()
                        )
                    pixels += x.numel()
                for side in missing:
                    cache[side][qs[side]] = counts[side] / pixels
            return tuple(cache[side][qs[side]] for side in (0, 1))

        fitted = paper_quantiles(risk, len(calibration), alpha=target)
    else:
        shape = (len(calibration), *calibration[0][0].shape)
        with tempfile.TemporaryDirectory(
            prefix="calibration-", dir=output
        ) as temporary:
            scores = [
                np.lib.format.open_memmap(
                    Path(temporary) / f"tail_{side}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                )
                for side in (0, 1)
            ]
            offset = 0
            try:
                for batch in loader:
                    y, x = batch[:, 0].to(device), batch[:, 1].to(device)
                    center, lo, hi = predict(y)
                    lower = (center - x) / (center - lo).clamp_min(1e-6)
                    upper = (x - center) / (hi - center).clamp_min(1e-6)
                    for dest, value in zip(scores, (lower, upper)):
                        dest[offset : offset + len(batch)] = value.cpu().numpy()
                    offset += len(batch)
                fitted = rcps_scales(
                    *scores, alpha=target, delta=settings()["calibration"]["delta"]
                )
            finally:
                for array in scores:
                    array._mmap.close()
        rng_after = capture_rng(device)
        # Replay the identical MC masks to audit the fitted calibration intervals.
        restore_rng(rng_before)
    covered, total, crossing = 0, 0, 0
    for batch in loader:
        y, x = batch[:, 0].to(device), batch[:, 1].to(device)
        _, lo, hi = apply_calibration(y, predict, q_model, run["method"], fitted)
        covered += int(((x >= lo) & (x <= hi)).sum())
        crossing += int((lo > hi).sum())
        total += x.numel()
    if rng_after is not None:
        restore_rng(rng_after)
    coverage = covered / total
    fitted.update(
        calibration_coverage=coverage,
        calibration_crossing=crossing / total,
        target_coverage=1 - target,
        n_images=len(calibration),
        calibrated_on="calibration split only",
        endpoint_clipping=None,
        has_hb_certificate="certificate" in fitted,
        calibration_mode="paper",
    )
    return fitted


def apply_calibration(y, predict, q_model, method, fitted):
    if method == "qutcc" and "lower_q" in fitted:
        return q_model.bounds(y, fitted["lower_q"], fitted["upper_q"])
    center, lo, hi = predict(y)
    if "scales" in fitted:
        lo = center - fitted["scales"]["lower"] * (center - lo).clamp_min(1e-6)
        hi = center + fitted["scales"]["upper"] * (hi - center).clamp_min(1e-6)
    return center, lo, hi


def run_evaluation(args, *, model_loader=None):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_evaluation(args, model_loader=model_loader)


@torch.inference_mode()
def _run_evaluation(args, *, model_loader=None):
    if args.calibration not in ("paper", "native"):
        raise ValueError("Calibration must be paper or native")
    source = source_fingerprint()
    software = environment()
    expected = (5, 10) if args.method == "ensemble" else (1,)
    if len(args.checkpoints) not in expected or args.workers < 0 or args.examples < 0:
        raise ValueError(
            "Expected one checkpoint, or five/ten ordered ensemble members"
        )
    main_identity = audit_manifest(args.manifest, args.dataset, smoke=args.smoke)
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
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(
        (out / name).exists()
        for name in ("report.json", "calibration.json", "failure.json")
    ):
        raise FileExistsError(
            "Use a new output folder; existing fitted results are immutable"
        )
    loaded, provenance = [], []
    started = time.monotonic()
    for index, path in enumerate(args.checkpoints):
        member_run = profile(
            args.experiment,
            args.dataset,
            args.method,
            index if args.method == "ensemble" else None,
        )
        model, info = (model_loader or load_model)(
            path, member_run, device, smoke=args.smoke
        )
        loaded.append(model)
        provenance.append(info)
    if len({row["epoch"] for row in provenance}) != 1:
        raise ValueError("Ensemble members must use the same completed epoch")
    if args.method == "ensemble":
        for field in ("initialization", "data"):
            if len({row["seeds"].get(field) for row in provenance}) != len(provenance):
                raise ValueError(
                    "Ensemble members need distinct initialization and data-order seeds"
                )
        from models import DeepEnsemble

        model = (
            DeepEnsemble(loaded, base_interval_alpha=settings()["alpha"])
            .to(device)
            .eval()
        )
    else:
        model = loaded[0]
    seed = (
        args.seed
        if args.seed is not None
        else (
            run["seeds"]["initialization"] if args.method == "mc_dropout" else 20261022
        )
    )
    torch.manual_seed(seed)
    load_seconds = time.monotonic() - started
    started = time.monotonic()
    perceptual = None if args.no_lpips else PerceptualMetric(device)
    lpips_load_seconds = time.monotonic() - started

    def predict(y):
        with autocast(run, device):
            result = (
                model.bounds(
                    y, samples=settings()["mc_samples"], alpha=settings()["alpha"]
                )
                if args.method == "mc_dropout"
                else model.bounds(y)
            )
        if not all(torch.isfinite(value).all() for value in result):
            raise ValueError("Nonfinite native predictions")
        return tuple(value.float() for value in result)

    request = dict(
        experiment=args.experiment,
        dataset=args.dataset,
        method=args.method,
        alpha=settings()["alpha"],
        calibration=args.calibration,
        seed=seed,
        mc_samples=settings()["mc_samples"] if args.method == "mc_dropout" else None,
        ensemble_members=len(loaded) if args.method == "ensemble" else None,
        device=str(device),
        smoke=args.smoke,
        test_used_for_calibration=False,
        lpips=not args.no_lpips,
    )
    fitted, cal_identity = {}, None
    started = time.monotonic()
    if args.calibration != "native":
        cal_identity = audit_manifest(
            args.calibration_manifest, args.dataset, smoke=args.smoke
        )
        cal_ids = set(split_ids(args.calibration_manifest, "calibration"))
        data = json.loads(args.manifest.read_text())
        for role in ("train", "val", "test"):
            if role in data.get("roles", data) and cal_ids.intersection(
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
                calibration,
                predict,
                model,
                run,
                device,
                args.workers,
                out,
            )
        except CalibrationFailure as error:
            write_json(
                out / "failure.json",
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
        out / "calibration.json", dict(fitted, request=request, data=cal_identity)
    )
    calibration_seconds = time.monotonic() - started
    # The test pixel loader is instantiated only after the calibration is frozen.
    test = dataset_for(args.dataset, args.manifest, args.pairs, "test")
    check_shape(test, args.dataset, smoke=args.smoke)
    rows, examples, offset = (
        [],
        {key: [] for key in ("measurement", "target", "prediction", "lower", "upper")},
        0,
    )
    started = time.monotonic()
    prediction_seconds, metric_seconds = 0.0, 0.0
    for batch in DataLoader(
        test, batch_size=run["microbatch"], num_workers=args.workers
    ):
        y, x = batch[:, 0].to(device), batch[:, 1].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_started = time.monotonic()
        prediction, lo, hi = apply_calibration(y, predict, model, args.method, fitted)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_seconds += time.monotonic() - prediction_started
        if not all(torch.isfinite(value).all() for value in (prediction, lo, hi)):
            raise ValueError("Nonfinite calibrated predictions")
        metric_started = time.monotonic()
        measured = {
            mode: interval_metrics(prediction, lo, hi, x, clipped=mode == "clipped")
            for mode in ("raw", "clipped")
        }
        for values in measured.values():
            for name, value in values.items():
                if (
                    name not in ("pearson", "spearman")
                    and not torch.isfinite(value).all()
                ):
                    raise ValueError("Nonfinite " + name + " scores")
        perceptual_values = (
            perceptual(prediction, x) if perceptual is not None else None
        )
        if (
            perceptual_values is not None
            and not torch.isfinite(perceptual_values).all()
        ):
            raise ValueError("Nonfinite LPIPS scores")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metric_seconds += time.monotonic() - metric_started
        for index in range(len(batch)):
            rows.append(
                dict(
                    index=offset + index,
                    id=test_ids[offset + index],
                    **{
                        mode: {
                            key: (
                                float(value[index])
                                if math.isfinite(float(value[index]))
                                else None
                            )
                            for key, value in metrics.items()
                        }
                        for mode, metrics in measured.items()
                    },
                )
            )
            rows[-1]["raw"][
                "lpips"
            ] = None  # Unbounded outputs are outside LPIPS's supported range.
            rows[-1]["clipped"]["lpips"] = (
                float(perceptual_values[index])
                if perceptual_values is not None
                else None
            )
            if offset + index < args.examples:
                for key, value in zip(
                    examples,
                    (y, x, prediction.clamp(0, 1), lo.clamp(0, 1), hi.clamp(0, 1)),
                ):
                    examples[key].append(value[index].cpu().numpy())
        offset += len(batch)
    aggregate = {
        mode: {
            key: finite_mean(row[mode][key] for row in rows) for key in rows[0][mode]
        }
        for mode in ("raw", "clipped")
    }
    metrics = {
        mode: {key: value[0] for key, value in values.items()}
        for mode, values in aggregate.items()
    }
    valid_counts = {
        mode: {key: value[1] for key, value in values.items()}
        for mode, values in aggregate.items()
    }
    report = dict(
        epoch=provenance[0]["epoch"],
        camera=args.dataset,
        method=args.method,
        request=request,
        test_count=len(rows),
        metrics=metrics,
        metric_valid_counts=valid_counts,
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
        data=dict(main=main_identity, calibration=cal_identity),
        environment=software,
        source_sha256=source,
        hardware=hardware(device),
        timing=dict(
            load_seconds=load_seconds,
            lpips_load_seconds=lpips_load_seconds,
            calibration_seconds=calibration_seconds,
            test_seconds=time.monotonic() - started,
            prediction_seconds=prediction_seconds,
            prediction_seconds_per_image=prediction_seconds / len(rows),
            metric_seconds=metric_seconds,
        ),
        per_image=rows,
        example_count=args.examples,
    )
    write_json(out / "report.json", report)
    with (out / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "index"]
            + [f"{mode}_{key}" for mode in ("raw", "clipped") for key in metrics[mode]],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(
                    id=row["id"],
                    index=row["index"],
                    **{
                        f"{mode}_{key}": value
                        for mode in ("raw", "clipped")
                        for key, value in row[mode].items()
                    },
                )
            )
    if args.examples:
        np.savez_compressed(
            out / "examples.npz",
            **{key: np.stack(value) for key, value in examples.items()},
        )
    print(json.dumps(dict(test_count=len(rows), **metrics["clipped"])), flush=True)
    return report


def main():
    run_evaluation(arguments())


if __name__ == "__main__":
    main()
