"""Recalibrate a frozen checkpoint near 90%, then evaluate the untouched test split.

Writes separate ``tight_empirical`` results. Does not replace RCPS certificates,
modify training, or use test targets to select scales/quantiles.
"""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from convrml.calibration import (
    CalibrationError,
    fit_tail_scale,
    search_quantiles,
)
from coverage90_cluster import validate_report

# Capture provenance before inference starts. Updating a source file while a
# long job runs must not attribute its old computation to a newer file version.
EVALUATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CALIBRATION_SHA256 = hashlib.sha256(
    (REPOSITORY / "convrml/calibration.py").read_bytes()
).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--source-folder",
        type=Path,
        required=True,
        help="Existing paper experiment folder, or baseline project root",
    )
    parser.add_argument("--family", choices=["paper", "baseline"], required=True)
    parser.add_argument("--camera", choices=["rml", "diffuser", "raw"], required=True)
    parser.add_argument(
        "--method",
        choices=[
            "qutcc",
            "im2im",
            "ensemble5",
            "ensemble10",
            "mc_005",
            "mc_010",
            "mc_020",
        ],
        required=True,
    )
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights-root",
        type=Path,
        help="Optional restored baseline checkpoint tree; never substitutes epochs",
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=0.001,
        help="Absolute coverage excess tolerance; .001 = 0.1 percentage point",
    )
    parser.add_argument("--examples", type=int, default=50)
    args = parser.parse_args(argv)
    if args.epoch <= 0 or args.examples < 0:
        parser.error("Invalid epoch/example count")
    if args.examples > (2000 if args.camera == "raw" else 5000):
        parser.error("Requested examples exceed the held-out test split")
    if not 0 < args.coverage_tolerance < 0.1:
        parser.error("Coverage tolerance must be in (0,.1)")
    if args.family == "baseline" and (args.camera == "raw" or args.method == "qutcc"):
        parser.error("Raw/QUTCC require the paper family")
    if args.family == "paper" and args.method not in ["qutcc", "im2im"]:
        parser.error("Paper family supports QUTCC and Im2im")
    if args.family == "paper" and args.weights_root:
        parser.error("--weights-root is for baseline checkpoints only")
    return args


def main(args):
    # Import original experiment modules only after choosing one import context.
    os.environ["CONVRML_PROJECT"] = str(args.project_root.resolve())
    os.chdir(args.project_root)
    sys.path.insert(0, str(args.source_folder.resolve()))
    import torch
    from torch.utils.data import DataLoader
    from torchmetrics.functional.image import (
        structural_similarity_index_measure as ssim,
    )

    torch.set_num_threads(4)
    torch.manual_seed(20261022)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    request = dict(
        family=args.family,
        camera=args.camera,
        method=args.method,
        epoch=args.epoch,
        source_folder=str(args.source_folder.resolve()),
        target_coverage=0.9,
        coverage_tolerance=args.coverage_tolerance,
        endpoint_clipping=[0, 1],
        calibration="tight_empirical",
        weights_root=str(args.weights_root.resolve()) if args.weights_root else None,
        test_used_for_calibration=False,
    )
    if (out / "report.json").exists():
        old = json.loads((out / "report.json").read_text())
        if old["request"] != request:
            raise ValueError("Refusing to overwrite results from a different request")
        task = dict(camera=args.camera, method=args.method, epoch=args.epoch)
        validate_report(task, old, required_examples=args.examples)
        if args.examples and not (out / "examples.npz").is_file():
            raise ValueError("Saved report is missing its example arrays")
        print("Already complete", out, flush=True)
        return

    initial = (0.05, 0.95)
    if args.family == "paper":
        from model import Model
        from data import Pairs, identity
        from train import PROTOCOL

        folder = args.source_folder / "results" / args.camera / args.method
        checkpoint = folder / "checkpoints" / f"epoch_{args.epoch:03d}.pt"
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            ck["epoch"] != args.epoch
            or ck["config"]["method"] != args.method
            or ck["config"]["camera"] != args.camera
            or ck["config"]["protocol"] != PROTOCOL
            or ck["config"]["dataset_identity"] != identity(args.camera)
        ):
            raise ValueError("Checkpoint/dataset/protocol identity mismatch")
        model = Model(args.method)
        model.load_state_dict(ck["model"], strict=True)
        model.cuda().eval().requires_grad_(False)
        micro = ck["config"]["microbatch"]
        split_identity = ck["config"]["dataset_identity"]
        provenance = [
            dict(
                checkpoint=str(checkpoint),
                epoch=ck["epoch"],
                protocol=PROTOCOL,
                source_sha256=ck["config"]["source_sha256"],
            )
        ]
        del ck
        cal = Pairs(args.camera, "calibration")
        make_test = lambda: Pairs(args.camera, "test")
        ids = lambda dataset: [str(row["id"]) for row in dataset.rows]
        old_fit = folder / "evaluation" / f"epoch_{args.epoch:03d}" / "calibration.json"
        if args.method == "qutcc" and old_fit.exists():
            saved = json.loads(old_fit.read_text())
            initial = (saved["lower_q"], saved["upper_q"])

        def predict(y, fitted=None):
            if args.method == "qutcc":
                qs = (
                    initial
                    if fitted is None
                    else (fitted["lower_q"], fitted["upper_q"])
                )
                return (
                    model(y),
                    model(y, y.new_full((len(y),), qs[0])),
                    model(y, y.new_full((len(y),), qs[1])),
                )
            z = model(y)
            return z[:, 1], z[:, 0], z[:, 2]

        autocast = nullcontext
    else:
        if args.weights_root:
            import scripts.corrected_comparison_study as study

            study.ROOT = args.weights_root.resolve()
        from scripts.export_corrected_milestone import predictor
        from scripts.corrected_comparison_study import paths
        from scripts.formal_rml_train import Pairs
        from scripts.training_records import manifest_digest

        predict_base, provenance = predictor(args.camera, args.method, args.epoch)
        data, caldir = paths(args.camera)
        # Reuse the immutable array directly, without constructing a training loader.
        array = np.load(caldir / "pairs.npy", mmap_mode="r")

        class Calibration(torch.utils.data.Dataset):
            def __len__(self):
                return len(array)

            def __getitem__(self, index):
                return torch.from_numpy(np.array(array[index], copy=True)).float() / 255

        cal = Calibration()
        make_test = lambda: Pairs("test", data)
        records = json.loads((data / "manifest.json").read_text())
        ids = lambda dataset: [str(row["id"]) for row in records["test"]]
        split_identity = dict(
            data=manifest_digest(data), calibration=manifest_digest(caldir)
        )
        micro = 2
        predict = lambda y, fitted=None: predict_base(y)
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    expected_cal = 3000 if args.camera == "raw" else 2048
    if len(cal) != expected_cal:
        raise ValueError("Wrong calibration split size")
    cal_loader = DataLoader(cal, batch_size=micro, num_workers=2, pin_memory=True)

    def move(batch):
        return batch[:, 0].cuda(non_blocking=True), batch[:, 1].cuda(non_blocking=True)

    def bounds(pred, lower, upper, fitted):
        if args.method != "qutcc":
            lower = pred - fitted["scales"]["lower"] * (pred - lower).clamp_min(1e-6)
            upper = pred + fitted["scales"]["upper"] * (upper - pred).clamp_min(1e-6)
        return lower, upper

    started = time.monotonic()
    with torch.inference_mode():
        if args.method == "qutcc":
            cache = [{}, {}]

            def risk(qlo, qhi):
                qs = (qlo, qhi)
                missing = [i for i in range(2) if qs[i] not in cache[i]]
                counts, pixels = [0, 0], 0
                if missing:
                    for batch in cal_loader:
                        y, x = move(batch)
                        for side in missing:
                            endpoint = model(y, y.new_full((len(y),), qs[side])).clamp(
                                0, 1
                            )
                            if not torch.isfinite(endpoint).all():
                                raise ValueError("Nonfinite calibration prediction")
                            counts[side] += int(
                                ((x < endpoint) if side == 0 else (x > endpoint)).sum()
                            )
                        pixels += x.numel()
                    for side in missing:
                        cache[side][qs[side]] = counts[side] / pixels
                values = tuple(cache[i][qs[i]] for i in range(2))
                print(
                    json.dumps(dict(stage="quantile_search", qs=qs, risks=values)),
                    flush=True,
                )
                return values

            fitted = search_quantiles(
                risk, initial=initial, coverage_tolerance=args.coverage_tolerance
            )
            # Audit actual joint coverage, including possible endpoint crossings.
            covered = total = crossing = 0
            for batch in cal_loader:
                y, x = move(batch)
                lo = model(y, y.new_full((len(y),), fitted["lower_q"])).clamp(0, 1)
                hi = model(y, y.new_full((len(y),), fitted["upper_q"])).clamp(0, 1)
                covered += int(((x >= lo) & (x <= hi)).sum())
                crossing += int((lo > hi).sum())
                total += x.numel()
            fitted.update(
                calibration_coverage=covered / total,
                calibration_crossing=crossing / total,
            )
        else:
            shape = (len(cal), *cal[0][0].shape)
            files = [out / "lower_scores.tmp.npy", out / "upper_scores.tmp.npy"]
            scores = [
                np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=shape)
                for f in files
            ]
            try:
                offset = 0
                for batch in cal_loader:
                    y, x = move(batch)
                    with autocast():
                        pred, lo, hi = predict(y)
                    pred, lo, hi = (v.float() for v in (pred, lo, hi))
                    if not all(torch.isfinite(v).all() for v in (pred, lo, hi)):
                        raise ValueError("Nonfinite calibration output")
                    ls = (pred - x) / (pred - lo).clamp_min(1e-6)
                    us = (x - pred) / (hi - pred).clamp_min(1e-6)
                    # A clipped lower bound cannot miss x=1, or upper miss x=0.
                    ls = ls.masked_fill(x == 1, -torch.inf)
                    us = us.masked_fill(x == 0, -torch.inf)
                    for score, value in zip(scores, (ls, us)):
                        score[offset : offset + len(y)] = value.cpu().numpy()
                    offset += len(y)
                    if offset % 256 == 0:
                        print(
                            "calibration predictions", offset, "/", len(cal), flush=True
                        )
                tails = [fit_tail_scale(score, overwrite=True) for score in scores]
                fitted = dict(
                    scales={
                        side: tail["scale"]
                        for side, tail in zip(("lower", "upper"), tails)
                    },
                    tail_risks={
                        side: tail["risk"]
                        for side, tail in zip(("lower", "upper"), tails)
                    },
                    calibration_coverage=1 - sum(tail["risk"] for tail in tails),
                    procedure="tight empirical asymmetric order-statistic scales; not RCPS",
                )
            finally:
                del scores
                # The loop variable also references the last memmap.
                if "score" in locals():
                    del score
                gc.collect()
                for path in files:
                    path.unlink(missing_ok=True)
        fitted.update(
            target_coverage=0.9,
            n_images=len(cal),
            coverage_tolerance=args.coverage_tolerance,
            within_tolerance=0.9 - 1e-12
            <= fitted["calibration_coverage"]
            <= 0.9 + args.coverage_tolerance,
            calibrated_on="calibration split only",
            delta=None,
            confidence_guarantee=False,
            endpoint_clipping=[0, 1],
        )
        # Commit calibration before even instantiating the test dataset.
        write(out / "calibration.json", fitted)
        calibration_seconds = time.monotonic() - started
        print("CALIBRATION", json.dumps(fitted), flush=True)
        test = make_test()
        identifiers = ids(test)
        if len(test) != (2000 if args.camera == "raw" else 5000):
            raise ValueError("Wrong held-out test split size")
        rows, arrays = [], {
            k: [] for k in ("measurement", "target", "prediction", "lower", "upper")
        }
        started = time.monotonic()
        for batch in DataLoader(test, batch_size=micro, num_workers=2, pin_memory=True):
            y, x = move(batch)
            with autocast():
                pred, lo, hi = predict(y, fitted)
            pred, lo, hi = (v.float() for v in (pred, lo, hi))
            lo, hi = bounds(pred, lo, hi, fitted)
            if not all(torch.isfinite(v).all() for v in (pred, lo, hi)):
                raise ValueError("Nonfinite test output")
            raw = (pred, lo, hi)
            clipped = tuple(v.clamp(0, 1) for v in raw)
            batch_metrics = {}
            for name, (p, l, u) in (("raw", raw), ("clipped", clipped)):
                error = p - x
                batch_metrics[name] = dict(
                    psnr=-10 * error.square().mean((1, 2, 3)).clamp_min(1e-12).log10(),
                    ssim=ssim(p, x, data_range=1.0, reduction="none"),
                    mae=error.abs().mean((1, 2, 3)),
                    coverage=((x >= l) & (x <= u)).float().mean((1, 2, 3)),
                    width=(u - l).mean((1, 2, 3)),
                    crossing=(l > u).float().mean((1, 2, 3)),
                )
            for j in range(len(y)):
                index = len(rows)
                rows.append(
                    dict(
                        index=index,
                        id=identifiers[index],
                        **{
                            name: {k: float(v[j]) for k, v in metrics.items()}
                            for name, metrics in batch_metrics.items()
                        },
                    )
                )
                if index < args.examples:
                    for key, value in zip(arrays, (y, x, *clipped)):
                        arrays[key].append(value[j].cpu().numpy())
            if len(rows) % 250 == 0:
                print("test", len(rows), "/", len(test), flush=True)
        if args.examples:
            np.savez_compressed(
                out / "examples.npz", **{k: np.stack(v) for k, v in arrays.items()}
            )
        aggregate = {
            name: {k: float(np.mean([r[name][k] for r in rows])) for k in rows[0][name]}
            for name in ("raw", "clipped")
        }
        write(
            out / "report.json",
            dict(
                request=request,
                calibration=fitted,
                test_count=len(rows),
                example_count=min(args.examples, len(rows)),
                metrics=aggregate,
                per_image=rows,
                provenance=provenance,
                split_identity=split_identity,
                calibration_seconds=calibration_seconds,
                inference_seconds=time.monotonic() - started,
                gpu=torch.cuda.get_device_name(),
                evaluator_sha256=EVALUATOR_SHA256,
                calibration_code_sha256=CALIBRATION_SHA256,
                notes="Checkpoint frozen. Empirical matched coverage, not RCPS. Test never selects endpoints. Raw and clipped metrics kept separately; uncertainty is upper-lower.",
            ),
        )
        print("COMPLETE", out, json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    args = arguments()
    try:
        main(args)
    except CalibrationError as error:
        write(
            args.output / "failure.json",
            dict(
                status="calibration_infeasible",
                camera=args.camera,
                method=args.method,
                epoch=args.epoch,
                error=str(error),
                error_code=error.code,
                test_used_for_calibration=False,
            ),
        )
        print("Calibration failed:", error, file=sys.stderr, flush=True)
        raise SystemExit(2)
