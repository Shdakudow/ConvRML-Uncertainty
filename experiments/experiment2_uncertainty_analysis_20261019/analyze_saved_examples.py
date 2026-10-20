"""Compute LPIPS and pixel correlations for exact saved Experiment 2 examples."""

import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from metrics import analyze, describe, correlation

BASE = Path.home() / ".local/share/convrml-epoch50-auto/results"
QBASE = Path.home() / ".local/share/convrml-paper50-demo-20261019"
METHODS = ["im2im", "ensemble5", "mc_005", "qutcc"]
LABELS = {
    "im2im": "Im2im",
    "ensemble5": "Ensemble (5)",
    "mc_005": "MC dropout (p=0.05, 30 passes)",
    "qutcc": "QUTCC",
}


def source(camera, method):
    return QBASE / camera if method == "qutcc" else BASE / camera / method


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none"
    ).eval()
    summary = {
        "scope": "First 50 test images in manifest order, matching demo; not the full 5000-image test set",
        "epoch": 50,
        "lpips": "LPIPS-AlexNet v0.1 via TorchMetrics; RGB [0,1], normalize=True",
        "maps": "error=mean_RGB(abs(prediction-ground_truth)); uncertainty=mean_RGB(upper-lower); clipped [0,1] predictions/endpoints",
        "primary": "Per-image unsmoothed spatial Pearson and Spearman, then mean across images",
        "smoothing": "Secondary Gaussian sigma=0.5,1,2,4 pixels, applied separately to each 2D map",
        "bootstrap": "2000 image-level resamples; 95% percentile CI for image means; no pixel-independence assumption",
        "cameras": {},
    }
    allrows = []
    for camera in ["rml", "diffuser"]:
        summary["cameras"][camera] = {}
        baseline = np.load(BASE / camera / "im2im/examples.npz")
        base_report = json.loads((BASE / camera / "im2im/report.json").read_text())
        ids = [str(r["id"]) for r in base_report["per_image"]]
        for method in METHODS:
            folder = source(camera, method)
            if (
                not (folder / "report.json").exists()
                or not (folder / "examples.npz").exists()
            ):
                summary["cameras"][camera][method] = {
                    "status": "pending_epoch50_export"
                }
                continue
            r = json.loads((folder / "report.json").read_text())
            assert r["epoch"] == 50
            report_ids = [str(v["id"]) for v in r["per_image"]]
            assert report_ids == ids
            cache = a.output / f"{camera}_{method}.json"
            fingerprint = hashlib.sha256(
                (folder / "report.json").read_bytes()
            ).hexdigest()
            if (
                cache.exists()
                and json.loads(cache.read_text()).get("source_report_sha256")
                == fingerprint
            ):
                rows = json.loads(cache.read_text())["rows"]
                print("Reuse", camera, method, flush=True)
            else:
                z = np.load(folder / "examples.npz")
                assert np.array_equal(
                    z["target"], baseline["target"]
                ) and np.array_equal(z["measurement"], baseline["measurement"])
                pred = np.clip(z["prediction"], 0, 1)
                gt = np.asarray(z["target"], dtype=np.float32)
                lo = np.clip(z["lower"], 0, 1)
                hi = np.clip(z["upper"], 0, 1)
                assert (
                    pred.shape == gt.shape == lo.shape == hi.shape == (50, 3, 256, 256)
                )
                rows = []
                with torch.inference_mode():
                    for i in range(0, 50, 5):
                        metric.reset()
                        values = (
                            metric(
                                torch.from_numpy(pred[i : i + 5]),
                                torch.from_numpy(gt[i : i + 5]),
                            )
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )
                        for j, value in enumerate(values):
                            index = i + j
                            rows.append(
                                {
                                    "camera": camera,
                                    "method": method,
                                    "id": ids[index],
                                    "index": index,
                                    "epoch": 50,
                                    "lpips": float(value),
                                    "scales": analyze(
                                        pred[index], gt[index], lo[index], hi[index]
                                    ),
                                }
                            )
                        print(camera, method, len(rows), "/50", flush=True)
                cache.write_text(
                    json.dumps(
                        {"source_report_sha256": fingerprint, "rows": rows}, indent=2
                    )
                    + "\n"
                )
            allrows.extend(rows)
            scales = {
                s: {
                    k: describe([v["scales"][s][k] for v in rows])
                    for k in ["pearson", "spearman", "mean_width", "mae"]
                }
                for s in ["0", "0.5", "1", "2", "4"]
            }
            means_u = np.array([v["scales"]["0"]["mean_width"] for v in rows])
            means_e = np.array([v["scales"]["0"]["mae"] for v in rows])
            summary["cameras"][camera][method] = {
                "status": "complete",
                "n_images": len(rows),
                "label": LABELS[method],
                "epoch": 50,
                "lpips": describe([v["lpips"] for v in rows]),
                "scales": scales,
                "image_level_width_vs_mae": correlation(means_u, means_e),
                "source_report_sha256": fingerprint,
            }
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (a.output / "per_image_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "camera",
                "method",
                "epoch",
                "id",
                "index",
                "lpips",
                "sigma",
                "pearson",
                "spearman",
                "mean_width",
                "mae",
                "n_valid_pixels",
            ],
        )
        writer.writeheader()
        for r in allrows:
            for sigma, values in r["scales"].items():
                writer.writerow(
                    {
                        **{k: v for k, v in r.items() if k != "scales"},
                        "sigma": sigma,
                        **values,
                    }
                )
    print("Saved", a.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
