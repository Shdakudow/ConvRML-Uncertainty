"""Check the frozen source, clean Mirflickr pairs and real-model training path."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch


def check_pair(row):
    """Verify both current files against the original manifest's byte/pixel hashes."""
    for field in ("measurement", "target"):
        expected = row[field]
        path = Path(expected["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["file_sha256"]:
            raise ValueError(f"Changed {field} bytes: {row['id']}")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        if pixels.shape != (270, 480, 3) or pixels.dtype != np.uint8:
            raise ValueError(f"Invalid RGB dimensions/type: {row['id']}")
        if hashlib.sha256(pixels.tobytes()).hexdigest() != expected["pixel_sha256"]:
            raise ValueError(f"Changed decoded {field}: {row['id']}")
        if np.all(pixels == pixels[0, 0]):
            raise ValueError(f"Constant {field}: {row['id']}")
    return row["id"]


def audit_data(manifest, output):
    from convrml.protocol import audit_manifest

    identity = audit_manifest(manifest, "raw_clean")
    value = json.loads(manifest.read_text())
    roles = value["roles"]
    # The existing split groups exact duplicates within a role. Check that this
    # remains true independently for measurement and target pixels.
    for field in ("measurement", "target"):
        seen = set()
        for role in ("train", "val", "test", "calibration"):
            hashes = {row[field]["pixel_sha256"] for row in roles[role]}
            if seen.intersection(hashes):
                raise ValueError(f"Duplicate {field} pixels across split roles")
            seen.update(hashes)
    rows = [
        row for role in ("train", "val", "test", "calibration") for row in roles[role]
    ]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as workers:
        for count, _ in enumerate(workers.map(check_pair, rows), start=1):
            if count % 1000 == 0:
                print(
                    json.dumps(dict(pairs_checked=count, total=len(rows))), flush=True
                )
    result = dict(
        passed=True,
        data=identity,
        pairs_checked=len(rows),
        current_file_and_pixel_hashes_match=True,
        duplicate_pixels_disjoint_across_roles=True,
        target_unchanged=True,
        seconds=time.monotonic() - started,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


def check_model(method, manifest, output):
    from convrml.protocol import (
        check_shape,
        dataset_for,
        environment,
        make_model,
        profile,
    )

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required for real-model preflight")
    run = profile(3, "raw_clean", method)
    training = dataset_for("raw_clean", manifest, None, "train")
    check_shape(training, "raw_clean")
    measurement, target = training[0].unsqueeze(0).to(device).unbind(dim=1)
    model = make_model(run).to(device).train()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Unexpected frozen training parameter")
    expected_shape = (1, 3, 270, 480)
    with torch.no_grad():
        bounds = model.bounds(measurement)
        if any(tuple(image.shape) != expected_shape for image in bounds):
            raise ValueError("Bounds were not cropped to original RGB dimensions")
        if not all(torch.isfinite(image).all() for image in bounds):
            raise ValueError("Nonfinite bounds")

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    quantile_rng = torch.Generator(device=device).manual_seed(run["seeds"]["quantile"])
    checks, losses = {}, []
    for step in range(2 if method == "qutcc" else 1):
        optimizer.zero_grad(set_to_none=True)
        if method == "qutcc":
            loss = model.loss(measurement, target, generator=quantile_rng)
        else:
            loss = model.loss(measurement, target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite model loss")
        loss.backward()
        gradients = {
            name: parameter.grad for name, parameter in model.named_parameters()
        }
        if any(
            not torch.isfinite(gradient).all()
            for gradient in gradients.values()
            if gradient is not None
        ):
            raise ValueError("Nonfinite gradients")
        for label, fragment in (
            ("encoder", "encoder.stem"),
            ("decoder", "refine.0"),
            ("head", ".head.weight"),
        ):
            checks[label + "_receives_gradients"] = any(
                fragment in name
                and gradient is not None
                and bool(gradient.abs().sum() > 0)
                for name, gradient in gradients.items()
            )
        if method == "im2im":
            head = model.backbone.head.weight.grad.reshape(
                3, 3, *model.backbone.head.weight.shape[1:]
            )
            checks["all_three_heads_receive_gradients"] = all(
                bool(part.abs().sum() > 0) for part in head
            )
        else:
            checks["film_sites"] = len(model.net.films) == 48
            checks["film_receives_gradients"] = any(
                ".films." in name
                and gradient is not None
                and bool(gradient.abs().sum() > 0)
                for name, gradient in gradients.items()
            )
            if step == 1:
                # Zero-initialized FiLM projections block the embedding gradient
                # on step one. It must reach the embedding after their first update.
                checks["embedding_receives_gradients_after_first_update"] = any(
                    "net.embedding." in name
                    and gradient is not None
                    and bool(gradient.abs().sum() > 0)
                    for name, gradient in gradients.items()
                )
        losses.append(float(loss.detach()))
        optimizer.step()
    if not all(checks.values()):
        raise ValueError("Failed gradient/FiLM checks: " + json.dumps(checks))
    if method == "qutcc":
        with torch.no_grad():
            low = model(measurement, measurement.new_full((1,), 0.1))
            high = model(measurement, measurement.new_full((1,), 0.9))
            checks["q_affects_predictions_after_update"] = not torch.equal(low, high)
        if not checks["q_affects_predictions_after_update"]:
            raise ValueError("Conditioning has no effect after gradient updates")
    if method == "im2im":
        del head
    else:
        del low, high
    del optimizer, model, gradients, loss, bounds, measurement, target
    gc.collect()
    torch.cuda.empty_cache()
    # Exercise the exact production trainer/Adam/microbatch/checkpoint path in a
    # temporary smoke directory. Production always starts from fresh weights.
    with tempfile.TemporaryDirectory(
        prefix=method + "-smoke-", dir=output.parent
    ) as temporary:
        command = [
            sys.executable,
            "train.py",
            "--dataset",
            "raw_clean",
            "--method",
            method,
            "--manifest",
            str(manifest),
            "--output",
            temporary,
            "--workers",
            "2",
            "--device",
            "cuda",
            "--smoke",
        ]
        subprocess.run(command, check=True)
        history = json.loads((Path(temporary) / "history.json").read_text())
    result = dict(
        passed=True,
        method=method,
        checks=checks,
        loss_steps=losses,
        smoke_history=history,
        gpu=torch.cuda.get_device_name(),
        environment=environment(),
        smoke_weights_used_for_production=False,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("data", "im2im", "qutcc"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "data":
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            check=True,
        )
        audit_data(args.manifest, args.output)
    else:
        check_model(args.action, args.manifest, args.output)


if __name__ == "__main__":
    main()
