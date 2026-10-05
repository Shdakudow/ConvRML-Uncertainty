#!/usr/bin/env python3
"""Render a small, reproducible evaluation report for the shared RML baseline."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from skimage.metrics import structural_similarity


DATA_ROOT = Path("/share/monakhova/shamus_data/parallel_100k")
MODEL_PATH = Path("/share/monakhova/shamus_data/convnext_models_20260917/rml.pt")
PROVENANCE_PATH = Path(
    "/share/monakhova/shamus_data/convnext_models_20260917/provenance/rml"
)
OUTPUT_DIR = Path("results/baseline_replication_20260918")

# First twelve IDs listed in the baseline's recorded held-out test split.
TEST_IDS = [12445, 48064, 66717, 51085, 40519, 8243, 91123, 18303, 13958, 26273, 17729, 86589]
DISPLAY_IDS = TEST_IDS[:6]


def find_single(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one file for {folder / pattern}, found {len(matches)}: {matches}")
    return matches[0]


def preprocess(path: Path) -> torch.Tensor:
    """Match the collaborator-provided inference preprocessing exactly."""
    with Image.open(path) as image:
        x = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255
    height, width = x.shape[-2:]
    scale = 256 / min(height, width)
    size = [max(256, round(height * scale)), max(256, round(width * scale))]
    x = F.interpolate(x[None], size=size, mode="bilinear", align_corners=False, antialias=True)
    top, left = [round((dimension - 256) / 2) for dimension in size]
    return x[:, :, top : top + 256, left : left + 256]


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((prediction - target) ** 2))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(prediction - target))),
        "psnr_db": float(10 * math.log10(1.0 / max(mse, np.finfo(np.float32).tiny))),
        "ssim": float(structural_similarity(target, prediction, channel_axis=-1, data_range=1.0)),
        "clipped_output_fraction": float(np.mean((prediction < 0.0) | (prediction > 1.0))),
    }


def as_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().permute(1, 2, 0).numpy()


def render_summary(pdf: PdfPages, recorded: dict, aggregate: dict, device: str) -> None:
    figure = plt.figure(figsize=(8.5, 11))
    figure.patch.set_facecolor("white")
    figure.text(0.08, 0.93, "ConvNeXtV2-Base RML replication", fontsize=22, weight="bold", color="#0b2545")
    figure.text(0.08, 0.895, "Shared baseline checkpoint | held-out subset verification", fontsize=11, color="#42647f")

    details = [
        ("Checkpoint", str(MODEL_PATH)),
        ("Dataset", "parallel_100k: rml → gt2rml"),
        ("Architecture", "ConvNeXtV2-Base, 95,277,315 parameters"),
        ("Input preprocessing", "RGB / 255; resize short side to 256; centered 256×256 crop"),
        ("Evaluation subset", f"{len(TEST_IDS)} listed held-out test IDs; no training images used"),
        ("Runtime", f"Torch {torch.__version__}; device {device}"),
    ]
    y = 0.835
    for label, value in details:
        figure.text(0.08, y, label, fontsize=10, weight="bold", color="#173b61")
        figure.text(0.27, y, value, fontsize=9.5, color="#202b35", wrap=True)
        y -= 0.043

    figure.text(0.08, 0.54, "Metrics", fontsize=15, weight="bold", color="#0b2545")
    axis = figure.add_axes([0.08, 0.34, 0.84, 0.16])
    axis.axis("off")
    table_rows = [
        ["Recorded full test", f"{recorded['psnr']:.3f}", f"{recorded['ssim']:.3f}", f"{recorded['mse']:.6f}", f"{recorded['mae']:.6f}", f"{int(recorded['count'])}"],
        ["This 12-image subset", f"{aggregate['psnr_db']:.3f}", f"{aggregate['ssim']:.3f}", f"{aggregate['mse']:.6f}", f"{aggregate['mae']:.6f}", str(len(TEST_IDS))],
    ]
    table = axis.table(
        cellText=table_rows,
        colLabels=["Evaluation", "PSNR (dB)", "SSIM", "MSE", "MAE", "Images"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.27, 0.15, 0.12, 0.16, 0.16, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d7e0e8")
        if row == 0:
            cell.set_facecolor("#0b4f6c")
            cell.set_text_props(color="white", weight="bold")
        elif row == 2:
            cell.set_facecolor("#eef6f8")

    note = (
        "Interpretation: the recorded score covers the baseline's full held-out test evaluation. "
        "The subset value above is a fresh execution of the exported TorchScript model on 12 IDs listed "
        "in its recorded test split. MSE, MAE, PSNR, and SSIM use the raw model output to match the original "
        "training evaluation; PNG panels clip predictions only for display."
    )
    figure.text(0.08, 0.26, note, fontsize=9.5, color="#34495e", wrap=True, linespacing=1.5)
    figure.text(
        0.08,
        0.09,
        f"Rendered {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | output directory: {OUTPUT_DIR}",
        fontsize=8,
        color="#6c7a89",
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def render_examples(pdf: PdfPages, examples: list[dict]) -> None:
    for offset in range(0, len(examples), 3):
        group = examples[offset : offset + 3]
        figure, axes = plt.subplots(len(group), 4, figsize=(8.5, 3.0 * len(group) + 0.8))
        if len(group) == 1:
            axes = np.expand_dims(axes, axis=0)
        figure.suptitle("Held-out RML examples", fontsize=17, weight="bold", color="#0b2545", y=0.98)
        headings = ["RML measurement", "Reconstruction", "Aligned target", "Absolute error"]
        for column, heading in enumerate(headings):
            axes[0, column].set_title(heading, fontsize=10, color="#173b61", pad=8)
        for row, example in enumerate(group):
            prediction = np.clip(example["prediction"], 0, 1)
            axes[row, 0].imshow(example["measurement"])
            axes[row, 1].imshow(prediction)
            axes[row, 2].imshow(example["target"])
            error = np.mean(np.abs(prediction - example["target"]), axis=2)
            axes[row, 3].imshow(error, cmap="magma", vmin=0, vmax=0.15)
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
            axes[row, 0].set_ylabel(
                f"ID {example['id']}\nPSNR {example['metrics']['psnr_db']:.2f} dB\nSSIM {example['metrics']['ssim']:.3f}",
                fontsize=8.5,
                color="#263746",
                rotation=0,
                labelpad=58,
                va="center",
            )
        figure.tight_layout(rect=(0.06, 0, 1, 0.95))
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (PROVENANCE_PATH / "metrics.json").open() as handle:
        recorded = json.load(handle)["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(str(MODEL_PATH), map_location=device).eval()

    records = []
    examples = []
    measurements = []
    targets = []
    paths = []
    for index in TEST_IDS:
        measurement_path = find_single(DATA_ROOT / "rml", f"4x_img_{index}_cam_*.tiff")
        target_path = find_single(DATA_ROOT / "gt2rml", f"warped_4x_undistorted_img_{index}_cam_*.tiff")
        measurements.append(preprocess(measurement_path))
        targets.append(preprocess(target_path))
        paths.append((measurement_path, target_path))

    with torch.inference_mode():
        predictions = model(torch.cat(measurements).to(device)).cpu()

    for index, measurement, target, prediction, (measurement_path, target_path) in zip(
        TEST_IDS, measurements, targets, predictions, paths
    ):
        measurement_np = as_image(measurement[0])
        target_np = as_image(target[0])
        prediction_np = as_image(prediction)
        metrics = image_metrics(prediction_np, target_np)
        record = {
            "id": index,
            "measurement": str(measurement_path),
            "target": str(target_path),
            **metrics,
        }
        records.append(record)
        np.save(OUTPUT_DIR / f"rml_{index}_prediction.npy", prediction_np)
        Image.fromarray((np.clip(prediction_np, 0, 1) * 255).round().astype(np.uint8)).save(
            OUTPUT_DIR / f"rml_{index}_prediction.png"
        )
        if index in DISPLAY_IDS:
            examples.append(
                {"id": index, "measurement": measurement_np, "target": target_np, "prediction": prediction_np, "metrics": metrics}
            )

    aggregate = {
        key: float(np.mean([record[key] for record in records]))
        for key in ("mse", "mae", "psnr_db", "ssim", "clipped_output_fraction")
    }
    with (OUTPUT_DIR / "subset_metrics.json").open("w") as handle:
        json.dump({"recorded_full_test": recorded, "replicated_subset": aggregate, "per_image": records}, handle, indent=2)
    with (OUTPUT_DIR / "subset_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    with PdfPages(OUTPUT_DIR / "baseline_replication.pdf") as pdf:
        render_summary(pdf, recorded, aggregate, device)
        render_examples(pdf, examples)

    print(json.dumps({"output_pdf": str(OUTPUT_DIR / "baseline_replication.pdf"), "replicated_subset": aggregate}, indent=2))


if __name__ == "__main__":
    main()
