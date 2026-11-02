"""Exp 4: fit on a shared held-out mixture; compare four frozen checkpoints.

Each model gets one global calibration, never one fit per test condition. Test
inputs are generated once per batch and shared by all four arms. Only the first
50 manifest-ordered test scenes are exported; metrics use all 2,000 scenes.
"""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from convrml.checkpoints import read_checkpoint, load_model_weights
from convrml.metrics import reconstruction_metrics
from dataset import ImagePairs
from models import QUTCC, Im2im
from experiments.experiment4_demo.conditions import (
    IDS,
    catalog,
    calibration_conditions,
    modify,
)
from experiments.experiment4_demo.empirical import fit_tail_scale, search_quantiles

MANIFEST_SHA = "155670232c27b3201582f5a1b67fb4196e8123e40f388d9c951df4418f7144fb"
CLEAN_SHA = dict(
    qutcc="297750d72e1b3c754ff23b4cbe3e8edc41e9f41800d3fa21d13b7418653edb0a",
    im2im="360551a7db56ca66c6aae29a0be975cad9a5165ccaa8ff2f76b887e906d35406",
)
INITIAL_SHA = dict(
    qutcc="4c0b9af045d9d13d0c90a2146fd712bcf1d3ee70455a153984419cf0f472da7d",
    im2im="98aff5f906eec2d1cfb836d9edb4a6bd969b09481083a84a92e814c1320392e3",
)
ARMS = ("qutcc_clean50", "qutcc_general100", "im2im_clean50", "im2im_general100")
STOP = 0


def stop(signum, _frame):
    global STOP
    STOP = 75 if signum == signal.SIGUSR1 else 1


def interrupted():
    if STOP:
        raise SystemExit(STOP)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def audit_manifest(path):
    if sha(path) != MANIFEST_SHA:
        raise ValueError("Original MIRFLICKR manifest changed")
    roles = json.loads(path.read_text())["roles"]
    seen = set()
    for name, count in dict(train=17999, val=2000, calibration=3000, test=2000).items():
        identifiers = {str(row["id"]) for row in roles[name]}
        if len(roles[name]) != count or len(identifiers) != count or seen & identifiers:
            raise ValueError("Wrong split size, duplicate IDs, or overlapping splits")
        seen.update(identifiers)


class ModifiedPairs(Dataset):
    """Repeatable inputs, with targets passed through unchanged."""

    def __init__(self, manifest, split, condition=None):
        self.pairs = ImagePairs(manifest, split)
        self.ids = self.pairs.ids
        self.split = split
        self.conditions = (
            calibration_conditions()
            if condition is None
            else (condition,) * len(self.pairs)
        )
        if len(self.conditions) != len(self.pairs):
            raise ValueError("Calibration mixture size mismatch")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        original, target = self.pairs[index]
        changed = modify(
            original[None],
            self.conditions[index],
            split=self.split,
            identifier=self.ids[index],
        )[0]
        return changed, target, original


def loader(dataset):
    return DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )


def move(batch):
    return batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)


def finite(*values):
    if not all(torch.isfinite(v).all() for v in values):
        raise ValueError("Nonfinite model output")


def load_arms(args):
    models, provenance = {}, {}
    for arm in ARMS:
        method, group = arm.split("_")
        general = group == "general100"
        epoch = 100 if general else 50
        folder = (
            args.generalized_root / "results" / method
            if general
            else args.clean_root / "results/raw" / method
        )
        path = folder / "checkpoints" / f"epoch_{epoch:03d}.pt"
        ck, digest = read_checkpoint(path)
        config = ck["config"]
        if ck["epoch"] != epoch or config["method"] != method or ck.get("smoke", False):
            raise ValueError(f"Wrong checkpoint identity: {arm}")
        if general:
            complete = json.loads((folder / "COMPLETE.json").read_text())
            if not (
                complete["passed"]
                and complete["epochs"] == 100
                and complete["method"] == method
            ):
                raise ValueError("Generalized training has not completed 100 epochs")
            if [row["epoch"] for row in ck["history"]] != list(range(1, 101)):
                raise ValueError("Incomplete generalized history")
            expected = dict(
                protocol="raw_online_ten_condition_from_scratch_20261031_v4_100epochs",
                manifest_sha256=MANIFEST_SHA,
                pretrained=False,
                initial_trained_steps=0,
                initial_weights_sha256=INITIAL_SHA[method],
                source_commit="4d94a20ca7b46ba9c8221e1800df27a5edfce1ed",
            )
            if any(config.get(k) != v for k, v in expected.items()):
                raise ValueError("Generalized training provenance changed")
        elif (
            digest != CLEAN_SHA[method]
            or config["protocol"] != "paper_qutcc_convnext_uniform_pinball_20261018_v1"
            or config["camera"] != "raw"
            or MANIFEST_SHA not in config["dataset_identity"].values()
        ):
            raise ValueError(
                "Baseline is not the original clean-trained demo checkpoint"
            )
        model = QUTCC() if method == "qutcc" else Im2im()
        load_model_weights(model, ck["model"])
        models[arm] = model.cuda().eval().requires_grad_(False)
        provenance[arm] = dict(
            method=method,
            training="mixed inputs" if general else "clean inputs",
            epochs=epoch,
            checkpoint=str(path),
            checkpoint_sha256=digest,
            training_config=config,
        )
        del ck
    return models, provenance


def predict(model, method, y, fit):
    if method == "qutcc":
        return model.bounds(y, fit["lower_q"], fit["upper_q"])
    p, lo, hi = model.bounds(y)
    return (
        p,
        p - fit["scales"]["lower"] * (p - lo).clamp_min(1e-6),
        p + fit["scales"]["upper"] * (hi - p).clamp_min(1e-6),
    )


@torch.inference_mode()
def audit_calibration(model, method, dataset, fit):
    covered = crossing = total = 0
    for batch in loader(dataset):
        interrupted()
        y, x = move(batch)
        p, lo, hi = predict(model, method, y, fit)
        finite(p, lo, hi)
        lo, hi = lo.clamp(0, 1), hi.clamp(0, 1)
        covered += int(((x >= lo) & (x <= hi)).sum())
        crossing += int((lo > hi).sum())
        total += x.numel()
    return dict(
        calibration_coverage=covered / total, calibration_crossing=crossing / total
    )


@torch.inference_mode()
def calibrate(model, arm, dataset, output):
    method = arm.split("_")[0]
    path = output / f"calibration/{arm}.json"
    if path.exists():
        return json.loads(path.read_text())
    if method == "qutcc":
        cache_path = output / f"calibration/{arm}-risks.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else [{}, {}]

        def risk(lo, hi):
            qs = (lo, hi)
            missing = [i for i in range(2) if str(qs[i]) not in cache[i]]
            counts, pixels = [0, 0], 0
            if missing:
                for batch in loader(dataset):
                    interrupted()
                    y, x = move(batch)
                    for side in missing:
                        endpoint = model(y, y.new_full((len(y),), qs[side])).clamp(0, 1)
                        finite(endpoint)
                        counts[side] += int(
                            ((x < endpoint) if side == 0 else (x > endpoint)).sum()
                        )
                    pixels += x.numel()
                for side in missing:
                    cache[side][str(qs[side])] = counts[side] / pixels
                write(cache_path, cache)
            values = tuple(cache[i][str(qs[i])] for i in range(2))
            print(
                json.dumps(
                    dict(stage="calibration_search", arm=arm, qs=qs, risks=values)
                ),
                flush=True,
            )
            return values

        # Probe feasibility first: failure is a meaningful robustness result.
        extreme = risk(0.0, 1.0)
        if max(extreme) > 0.05:
            fit = dict(
                lower_q=0.0,
                upper_q=1.0,
                status="target_unattainable",
                failure="The extreme q=0/1 endpoints miss more than 5% in at least one tail.",
            )
        else:
            try:
                fit = dict(search_quantiles(risk), status="fitted")
            except ValueError as error:
                # Do not label observed nonmonotonicity as successful calibration.
                fit = dict(
                    lower_q=0.0, upper_q=1.0, status="search_failed", failure=str(error)
                )
        fit["procedure"] = (
            "empirical equal-tail q search; endpoints 0 and 1 if search fails; not RCPS"
        )
    else:
        shape = (len(dataset), 3, 270, 480)
        files = [output / f"{arm}-{tail}-scores.tmp.npy" for tail in ("lower", "upper")]
        scores = [
            np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=shape)
            for f in files
        ]
        try:
            offset = 0
            for batch in loader(dataset):
                interrupted()
                y, x = move(batch)
                p, lo, hi = model.bounds(y)
                finite(p, lo, hi)
                lower = ((p - x) / (p - lo).clamp_min(1e-6)).masked_fill(
                    x == 1, -torch.inf
                )
                upper = ((x - p) / (hi - p).clamp_min(1e-6)).masked_fill(
                    x == 0, -torch.inf
                )
                scores[0][offset : offset + len(y)] = lower.cpu().numpy()
                scores[1][offset : offset + len(y)] = upper.cpu().numpy()
                offset += len(y)
            interrupted()
            tails = [fit_tail_scale(score, overwrite=True) for score in scores]
            fit = dict(
                scales=dict(zip(("lower", "upper"), (v["scale"] for v in tails))),
                status="fitted",
                procedure="empirical asymmetric order-statistic scales; not RCPS",
            )
        finally:
            for score in scores:
                score.flush()
                score._mmap.close()
            for file in files:
                file.unlink(missing_ok=True)
    fit.update(audit_calibration(model, method, dataset, fit))
    fit.update(
        target_coverage=0.90,
        n_images=len(dataset),
        within_tolerance=0.90 - 1e-12 <= fit["calibration_coverage"] <= 0.901,
        target_met=fit["calibration_coverage"] >= 0.90 - 1e-12,
        calibrated_on="shared calibration split mixture only",
        confidence_guarantee=False,
        test_used_for_calibration=False,
        endpoint_clipping=[0, 1],
    )
    write(path, fit)
    return fit


def save_rgb(path, tensor):
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    pixels = np.floor(array * 255 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def save_width(path, lower, upper):
    lo, hi = (v.detach().cpu().numpy() for v in (lower, upper))
    width = np.mean(hi - lo, axis=0)
    stops = np.array(
        [[20, 35, 60], [34, 95, 157], [35, 185, 172], [243, 203, 85], [250, 112, 82]]
    )
    t = np.clip(width, 0, 1) * 4
    index = np.minimum(t.astype(int), 3)
    rgb = stops[index] + (stops[index + 1] - stops[index]) * (t - index)[..., None]
    rgb = np.floor(rgb + 0.5).astype(np.uint8)
    rgb[np.any(lo > hi, axis=0)] = (227, 32, 201)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def metrics(p, lo, hi, x):
    p, lo, hi = (v.clamp(0, 1) for v in (p, lo, hi))
    result = reconstruction_metrics(p, x)
    result.update(
        coverage=((x >= lo) & (x <= hi)).float().mean((1, 2, 3)),
        width=(hi - lo).mean((1, 2, 3)),
        crossing=(lo > hi).float().mean((1, 2, 3)),
    )
    return {key: float(value[0]) for key, value in result.items()}


def parity(rows, reference, method, identifiers):
    expected = reference[method]
    for index in range(50):
        if expected[index]["id"] != identifiers[index]:
            raise ValueError("Baseline example ordering differs from the current demo")
        for key, tolerance in dict(psnr=0.001, ssim=5e-5, mae=2e-5).items():
            if abs(rows[index][key] - expected[index]["clipped"][key]) > tolerance:
                raise ValueError(
                    f"Clean baseline parity failed: {method}, {index}, {key}"
                )


@torch.inference_mode()
def evaluate_condition(models, fits, args, condition, web, reference):
    report_path = web / f"reports/{condition}.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    test = ModifiedPairs(args.manifest, "test", condition)
    if len(test) != 2000:
        raise ValueError("Wrong test count")
    all_rows = {arm: [] for arm in ARMS}
    digest = hashlib.sha256()
    for index, batch in enumerate(loader(test)):
        interrupted()
        digest.update(batch[0].numpy().tobytes())
        y, x = move(batch)
        if index < 50:
            if condition == "clean":
                save_rgb(web / f"images/shared/{index:03d}-original.png", batch[2][0])
                save_rgb(web / f"images/shared/{index:03d}-target.png", x[0])
            else:
                save_rgb(web / f"images/{condition}/{index:03d}-input.png", y[0])
        for arm, model in models.items():
            p, lo, hi = predict(model, arm.split("_")[0], y, fits[arm])
            finite(p, lo, hi)
            all_rows[arm].append(metrics(p, lo, hi, x))
            if index < 50:
                save_rgb(
                    web / f"images/{condition}/{arm}/{index:03d}-prediction.png", p[0]
                )
                save_width(
                    web / f"images/{condition}/{arm}/{index:03d}-width.png",
                    lo[0].clamp(0, 1),
                    hi[0].clamp(0, 1),
                )
        if (index + 1) % 100 == 0:
            write(
                args.output / "progress.json",
                dict(
                    stage="test", condition=condition, processed=index + 1, total=2000
                ),
            )
            print(condition, index + 1, "/2000", flush=True)
    if condition == "clean":
        for method in ("qutcc", "im2im"):
            parity(all_rows[f"{method}_clean50"], reference, method, test.ids)
    report = dict(
        condition=condition,
        count=len(test),
        modified_input_float32_sha256=digest.hexdigest(),
        shared_inputs_across_arms=True,
        arms={
            arm: dict(
                mean={key: float(np.mean([r[key] for r in rows])) for key in rows[0]},
                examples=rows[:50],
                per_image=rows,
            )
            for arm, rows in all_rows.items()
        },
    )
    write(report_path, report)
    return report


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    torch.set_num_threads(2)
    torch.manual_seed(20261101)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    audit_manifest(args.manifest)
    models, provenance = load_arms(args)
    source = hashlib.sha256()
    source.update((REPOSITORY / "dataset.py").read_bytes())
    for folder in (
        "models",
        "convrml",
        "input_modifications",
        "experiments/experiment4_demo",
    ):
        for path in sorted((REPOSITORY / folder).rglob("*.py")):
            source.update(str(path.relative_to(REPOSITORY)).encode())
            source.update(path.read_bytes())
    request = dict(
        schema="experiment4-v1",
        manifest_sha256=MANIFEST_SHA,
        evaluator_source_sha256=source.hexdigest(),
        baseline_reference_sha256=sha(args.baseline_reference),
        arms=provenance,
        calibration_mixture=dict(Counter(calibration_conditions())),
        test_count_per_condition=2000,
        examples=50,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    request_path = args.output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError(
            "Refusing to reuse outputs with different checkpoint/source/data identity"
        )
    write(request_path, request)
    cal = ModifiedPairs(args.manifest, "calibration")
    fits = {}
    for arm, model in models.items():
        write(args.output / "progress.json", dict(stage="calibration", arm=arm))
        fits[arm] = calibrate(model, arm, cal, args.output)
    # All fits are committed before opening test images. Never tune with test data.
    web = args.output / "web_data"
    reference = json.loads(args.baseline_reference.read_text())
    reports = {
        condition: evaluate_condition(models, fits, args, condition, web, reference)
        for condition in IDS
    }
    for arm, row in provenance.items():
        if sha(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during evaluation")
    test = ImagePairs(args.manifest, "test")
    data = dict(
        schema="experiment4-v1",
        title="Exp 4 · Generalized inputs",
        base="./",
        ids=list(test.ids[:50]),
        arms=provenance,
        calibration=fits,
        conditions=catalog(),
        results={
            c: dict(
                count=r["count"],
                modified_input_float32_sha256=r["modified_input_float32_sha256"],
                arms={
                    a: dict(mean=v["mean"], examples=v["examples"])
                    for a, v in r["arms"].items()
                },
            )
            for c, r in reports.items()
        },
        protocol=request,
        heatmap=dict(min=0, max=1, crossed_color="#e320c9"),
        metrics_clipped=True,
    )
    write(web / "data.json", data)
    files = {
        str(p.relative_to(web)): dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(web.rglob("*"))
        if p.is_file() and p.name != "READY.json"
    }
    if sum(name.endswith(".png") for name in files) != 5000:
        raise ValueError("Expected exactly 5,000 display images")
    write(
        web / "READY.json",
        dict(
            schema="experiment4-v1",
            passed=True,
            files=files,
            request_sha256=sha(request_path),
            completed=time.time(),
        ),
    )
    write(args.output / "progress.json", dict(stage="ready", passed=True))
    print("READY", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "generalized-root",
        "clean-root",
        "baseline-reference",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    main(parser.parse_args())
