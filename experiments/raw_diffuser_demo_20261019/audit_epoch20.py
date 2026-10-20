"""Recheck epoch-20 calibration; export honest native intervals if infeasible."""

import json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

SOURCE = Path.home() / "ConvRML/experiments/qutcc_paper_style_20261018"
sys.path.insert(0, str(SOURCE))
from model import Model
from data import Pairs, identity
from calibration import search_quantiles, tail_budget, CalibrationFailure

ROOT = Path.home() / "ConvRML/experiments/raw_diffuser_demo_20261019"


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    ck = torch.load(
        SOURCE / "results/raw/qutcc/checkpoints/epoch_020.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ck["epoch"] == 20 and ck["config"]["dataset_identity"] == identity("raw")
    model = Model("qutcc")
    model.load_state_dict(ck["model"], strict=True)
    model = model.cuda().eval()
    del ck
    cal = Pairs("raw", "calibration")
    trace = []

    def risk(lo, hi):
        low = high = pixels = 0
        for b in DataLoader(cal, batch_size=1, num_workers=2):
            y, x = b[:, 0].cuda(), b[:, 1].cuda()
            l = model(y, y.new_full((1,), lo))
            u = model(y, y.new_full((1,), hi))
            low += int((x < l).sum())
            high += int((x > u).sum())
            pixels += x.numel()
        result = (low / pixels, high / pixels)
        trace.append(
            dict(lower_q=lo, upper_q=hi, lower_risk=result[0], upper_risk=result[1])
        )
        print(json.dumps(trace[-1]), flush=True)
        return result

    failed = None
    try:
        calibration = search_quantiles(risk, len(cal))
        status = "calibrated"
    except CalibrationFailure as e:
        failed = str(e)
        calibration = dict(
            lower_q=0.05,
            upper_q=0.95,
            calibrated=False,
            procedure="Uncalibrated native quantile reference only",
        )
        status = "native_only"
    rows = []
    test = Pairs("raw", "test")
    for i, b in enumerate(DataLoader(test, batch_size=1, num_workers=2)):
        y, x = b[:, 0].cuda(), b[:, 1].cuda()
        pred = model(y)
        lo = model(y, y.new_full((1,), calibration["lower_q"]))
        hi = model(y, y.new_full((1,), calibration["upper_q"]))
        row = dict(index=i, id=test.rows[i]["id"])
        for label, p, l, u in [
            ("raw", pred, lo, hi),
            ("clipped", pred.clamp(0, 1), lo.clamp(0, 1), hi.clamp(0, 1)),
        ]:
            row[label] = dict(
                psnr=float(-10 * (p - x).square().mean().clamp_min(1e-12).log10()),
                ssim=float(ssim(p, x, data_range=1.0)),
                mae=float((p - x).abs().mean()),
                coverage=float(((x >= l) & (x <= u)).float().mean()),
                width=float((u - l).mean()),
                crossing=float((l > u).float().mean()),
            )
        rows.append(row)
    metrics = {
        label: {
            k: sum(row[label][k] for row in rows) / len(rows) for k in rows[0][label]
        }
        for label in ("raw", "clipped")
    }
    out = dict(
        epoch=20,
        method="qutcc",
        study="clean",
        status=status,
        calibrated=status == "calibrated",
        calibration=calibration,
        calibration_failure=failed,
        calibration_trace=trace,
        per_tail_target=tail_budget(0.1, len(cal)),
        test_count=len(rows),
        metrics=metrics,
        per_image=rows,
        gpu=torch.cuda.get_device_name(),
        note="Native reference is shown separately from calibrated intervals; never interpolated or relabeled as calibrated.",
    )
    path = ROOT / "exports/epoch20_reference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)
    print(json.dumps({k: v for k, v in out.items() if k != "per_image"}), flush=True)


if __name__ == "__main__":
    main()
