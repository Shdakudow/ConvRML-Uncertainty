#!/usr/bin/env python3
"""Render held-out reconstructions and quantile-interval maps for a pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import structural_similarity

from dataset import ScalableDataset
from models.convnext import QuantileConvRecon
from utils import crop_borders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    return parser.parse_args()


def to_image(tensor):
    return tensor.detach().cpu().permute(1, 2, 0).numpy()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantileConvRecon(
        n_channels=config.model.n_channels,
        height=config.model.output_height,
        width=config.model.output_width,
        model_size=config.model.size,
        drop_path_rate=config.model.drop_path_rate,
        layer_scale_init_value=config.model.layer_scale_init_value,
        decoder_channels=config.model.decoder_channels,
        quantile_embedding_dim=config.quantile.embedding_dim,
        quantile_condition_channels=config.quantile.condition_channels,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ScalableDataset(
        config.dataset.data_path,
        config.dataset.name,
        config.model.input_height,
        config.model.input_width,
        config.dataset.size,
        config.dataset.downsize_factor,
        train_type="test",
        use_processed=config.dataset.use_processed,
    )
    samples = []
    with torch.inference_mode():
        for index in range(min(args.num_samples, len(dataset))):
            measurement, target, image_id = dataset[index]
            measurement_batched = measurement[None].to(device)
            q10 = torch.tensor([0.1], device=device)
            q50 = torch.tensor([0.5], device=device)
            q90 = torch.tensor([0.9], device=device)
            reconstruction, lower = model(measurement_batched, q10)
            _, median = model(measurement_batched, q50)
            _, upper = model(measurement_batched, q90)
            reconstruction, lower, cropped_target = (
                reconstruction[0], lower[0], target.to(device)
            )
            upper, median = upper[0], median[0]
            cropped_reconstruction, cropped_target = crop_borders((reconstruction[None], cropped_target[None]), config, batch=True)
            mse = float(torch.mean((cropped_reconstruction[0] - cropped_target[0]) ** 2))
            psnr = 10 * math.log10(1.0 / max(mse, np.finfo(np.float32).tiny))
            reconstruction_np = to_image(reconstruction)
            target_np = to_image(target)
            lower_np, median_np, upper_np = to_image(lower), to_image(median), to_image(upper)
            interval = upper_np - lower_np
            crossing = float(np.mean(interval < 0.0))
            interval_width = np.mean(np.clip(interval, 0.0, None), axis=2)
            ssim = float(structural_similarity(
                to_image(cropped_target[0]), to_image(cropped_reconstruction[0]), channel_axis=-1, data_range=1.0
            ))
            samples.append({
                "id": image_id,
                "measurement": to_image(measurement),
                "reconstruction": reconstruction_np,
                "target": target_np,
                "q10": lower_np,
                "q50": median_np,
                "q90": upper_np,
                "interval_width": interval_width,
                "psnr_db": psnr,
                "ssim": ssim,
                "crossing_fraction": crossing,
            })

    vmax = max(float(np.quantile(np.concatenate([item["interval_width"].ravel() for item in samples]), 0.99)), 1e-4)
    report_rows = []
    pdf_path = output_dir / "quantile_comparison_and_uncertainty.pdf"
    with PdfPages(pdf_path) as pdf:
        for sample in samples:
            figure, axes = plt.subplots(1, 4, figsize=(12, 3.6))
            figure.suptitle(
                f"Held-out RML sample {sample['id']} | reconstruction PSNR {sample['psnr_db']:.2f} dB | SSIM {sample['ssim']:.3f}",
                fontsize=12,
                weight="bold",
            )
            panels = [
                ("RML measurement", sample["measurement"], None),
                ("MSE reconstruction", np.clip(sample["reconstruction"], 0, 1), None),
                ("Aligned target", np.clip(sample["target"], 0, 1), None),
                ("Q0.9 − Q0.1 interval width", sample["interval_width"], "magma"),
            ]
            for axis, (title, image, cmap) in zip(axes, panels):
                artist = axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=vmax if cmap else None)
                axis.set_title(title, fontsize=10)
                axis.set_xticks([])
                axis.set_yticks([])
                if cmap:
                    colorbar = figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
                    colorbar.ax.set_ylabel("predicted interval width", fontsize=8)
            figure.text(
                0.5, 0.02,
                f"Positive interval width shown; q-crossing fraction before clipping: {sample['crossing_fraction']:.2%}. "
                "This map is a learned quantile interval proxy, not calibrated epistemic uncertainty.",
                ha="center", fontsize=8.5, color="#44546a",
            )
            figure.tight_layout(rect=(0, 0.07, 1, 0.92))
            png_path = output_dir / f"comparison_{sample['id']}.png"
            figure.savefig(png_path, dpi=180, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            report_rows.append({key: sample[key] for key in ("id", "psnr_db", "ssim", "crossing_fraction")})

    with (output_dir / "summary.json").open("w") as handle:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "samples": report_rows,
            "mean_psnr_db": float(np.mean([row["psnr_db"] for row in report_rows])),
            "mean_ssim": float(np.mean([row["ssim"] for row in report_rows])),
            "mean_q_crossing_fraction": float(np.mean([row["crossing_fraction"] for row in report_rows])),
        }, handle, indent=2)
    print(json.dumps({"pdf": str(pdf_path), "mean_psnr_db": float(np.mean([row["psnr_db"] for row in report_rows])), "mean_ssim": float(np.mean([row["ssim"] for row in report_rows]))}, indent=2))


if __name__ == "__main__":
    main()
