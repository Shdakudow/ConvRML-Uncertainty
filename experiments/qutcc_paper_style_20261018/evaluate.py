"""Freeze a saved milestone, calibrate on the calibration split, then test once."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from model import Model
from data import Pairs, identity
from train import ROOT, TASKS, PROTOCOL, write
from calibration import search_quantiles, tail_budget, CalibrationFailure


@torch.inference_mode()
def run(a):
    camera, method = TASKS[a.task]
    folder = ROOT / "results" / camera / method
    out = folder / "evaluation" / f"epoch_{a.epoch:03d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        return
    ck = torch.load(
        folder / "checkpoints" / f"epoch_{a.epoch:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ck["config"]["protocol"] == PROTOCOL and ck["config"][
        "dataset_identity"
    ] == identity(camera)
    model = Model(method)
    model.load_state_dict(ck["model"], strict=True)
    model = model.cuda().eval()
    model.requires_grad_(False)
    config = ck["config"]
    del ck
    torch.set_num_threads(2)
    micro = config["microbatch"]
    cal = Pairs(camera, "calibration")
    loader = DataLoader(cal, batch_size=micro, num_workers=2)
    t = time.monotonic()
    if method == "qutcc":

        def risk(qlo, qhi):
            below = above = pixels = 0
            for b in loader:
                y, x = b[:, 0].cuda(), b[:, 1].cuda()
                lo = model(y, y.new_full((len(y),), qlo))
                hi = model(y, y.new_full((len(y),), qhi))
                assert torch.isfinite(lo).all() and torch.isfinite(hi).all()
                below += int((x < lo).sum())
                above += int((x > hi).sum())
                pixels += x.numel()  # Include all RGB channels, exclude padding.
            result = (below / pixels, above / pixels)
            print(json.dumps(dict(qlo=qlo, qhi=qhi, risks=result)), flush=True)
            return result

        fitted = search_quantiles(risk, len(cal))
    else:
        # Paper's asymmetric Im2im baseline: retain independent output scales.
        target = tail_budget(0.1, len(cal))
        low_scores = []
        high_scores = []
        for b in loader:
            y, x = b[:, 0].cuda(), b[:, 1].cuda()
            z = model(y)
            center = z[:, 1]
            low_scores.append(
                ((center - x) / (center - z[:, 0]).clamp_min(1e-6))
                .flatten()
                .cpu()
                .numpy()
            )
            high_scores.append(
                ((x - center) / (z[:, 2] - center).clamp_min(1e-6))
                .flatten()
                .cpu()
                .numpy()
            )
        # Scores are processed one side at a time to limit host memory. A
        # 0..6 / 1000 reference scale grid is searched via the actual risk.
        fits = {}
        for name, chunks in [("lower", low_scores), ("upper", high_scores)]:
            scores = np.concatenate(chunks)
            grid = np.linspace(0, 6, 1000)
            left, right = 0, len(grid) - 1
            if float(np.mean(scores > grid[-1])) > target:
                raise CalibrationFailure(
                    "No feasible Im2im scale in reference range 0..6"
                )
            while left < right:
                mid = (left + right) // 2
                if float(np.mean(scores > grid[mid])) <= target:
                    right = mid
                else:
                    left = mid + 1
            fits[name] = float(grid[left])
            del scores
        fitted = dict(
            scales=fits,
            alpha=0.1,
            per_tail_target=target,
            n_images=len(cal),
            procedure="paper asymmetric Im2im adjusted empirical tail-risk scale search",
            calibrated_on="calibration split only",
        )
    cal_seconds = time.monotonic() - t
    write(out / "calibration.json", fitted)
    test = Pairs(camera, "test")
    rows = []
    examples = {}
    t = time.monotonic()
    for b in DataLoader(test, batch_size=micro, num_workers=2):
        y, x = b[:, 0].cuda(), b[:, 1].cuda()
        if method == "qutcc":
            pred = model(y)
            lo = model(y, y.new_full((len(y),), fitted["lower_q"]))
            hi = model(y, y.new_full((len(y),), fitted["upper_q"]))
        else:
            z = model(y)
            pred = z[:, 1]
            lo = pred - fitted["scales"]["lower"] * (pred - z[:, 0]).clamp_min(1e-6)
            hi = pred + fitted["scales"]["upper"] * (z[:, 2] - pred).clamp_min(1e-6)
        assert all(torch.isfinite(v).all() for v in [pred, lo, hi])
        for j in range(len(y)):
            idx = len(rows)
            p, l, u, target = (
                pred[j : j + 1],
                lo[j : j + 1],
                hi[j : j + 1],
                x[j : j + 1],
            )
            row = dict(index=idx, id=test.rows[idx]["id"])
            for label, pp, ll, uu in [
                ("raw", p, l, u),
                ("clipped", p.clamp(0, 1), l.clamp(0, 1), u.clamp(0, 1)),
            ]:
                row[label] = dict(
                    psnr=float(
                        -10 * (pp - target).square().mean().clamp_min(1e-12).log10()
                    ),
                    ssim=float(ssim(pp, target, data_range=1.0)),
                    mae=float((pp - target).abs().mean()),
                    coverage=float(((target >= ll) & (target <= uu)).float().mean()),
                    width=float((uu - ll).mean()),
                    crossing=float((ll > uu).float().mean()),
                )
            rows.append(row)
            if idx < 10:
                for name, v in [
                    ("measurement", y[j]),
                    ("target", x[j]),
                    ("prediction", p[0]),
                    ("lower", l[0]),
                    ("upper", u[0]),
                ]:
                    examples[f"{idx}_{name}"] = v.cpu().numpy()
    np.savez_compressed(out / "examples.npz", **examples)
    aggregate = {
        label: {k: float(np.mean([r[label][k] for r in rows])) for k in rows[0][label]}
        for label in ("raw", "clipped")
    }
    write(
        out / "report.json",
        dict(
            protocol=PROTOCOL,
            camera=camera,
            method=method,
            epoch=a.epoch,
            config=config,
            calibration=fitted,
            test_count=len(rows),
            example_count=10,
            metrics=aggregate,
            per_image=rows,
            calibration_seconds=cal_seconds,
            inference_seconds=time.monotonic() - t,
            gpu=torch.cuda.get_device_name(),
            notes="No learned q selector, width loss, or median anchoring. Calibration uses raw predictions; both raw and clipped test metrics reported. Crossing is measured, never silently sorted.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, choices=range(4), required=True)
    p.add_argument("--epoch", type=int, required=True)
    args = p.parse_args()
    try:
        run(args)
    except CalibrationFailure as e:
        c, m = TASKS[args.task]
        write(
            ROOT
            / "results"
            / c
            / m
            / "evaluation"
            / f"epoch_{args.epoch:03d}"
            / "failure.json",
            dict(calibrated=False, error=str(e)),
        )
        print("Calibration infeasible at this milestone:", e, flush=True)
    except Exception as e:
        c, m = TASKS[args.task]
        write(
            ROOT
            / "results"
            / c
            / m
            / "evaluation"
            / f"epoch_{args.epoch:03d}"
            / "failure.json",
            dict(error=str(e)),
        )
        raise
