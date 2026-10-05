#!/usr/bin/env python3
"""Train a reconstruction head and a FiLM-conditioned quantile head."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm

from dataset import get_loader
from models.convnext import QuantileConvRecon
from utils import crop_borders


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional RML quantile reconstructions")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Cap steps per epoch for a smoke test")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Cap validation steps for a smoke test")
    return parser.parse_args()


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: torch.Tensor) -> torch.Tensor:
    """Elementwise quantile regression (pinball) loss for q in (0, 1)."""
    q = quantile.reshape(-1, 1, 1, 1).to(dtype=prediction.dtype)
    residual = target - prediction
    return torch.maximum(q * residual, (q - 1.0) * residual).mean()


def crop_outputs(reconstruction, quantile_reconstruction, target, config):
    if config.modes.full_fov_loss:
        return reconstruction, quantile_reconstruction, target
    full_target = target
    reconstruction, target = crop_borders((reconstruction, full_target), config, batch=True)
    quantile_reconstruction, _ = crop_borders((quantile_reconstruction, full_target), config, batch=True)
    return reconstruction, quantile_reconstruction, target


def combined_loss(reconstruction, quantile_reconstruction, target, quantile, config, crossing=None):
    mse = F.mse_loss(reconstruction, target)
    pinball = pinball_loss(quantile_reconstruction, target, quantile)
    crossing = torch.zeros((), device=target.device) if crossing is None else crossing
    crossing_weight = float(OmegaConf.select(config, "loss.crossing_weight", default=0.0))
    total = (
        config.loss.reconstruction_weight * mse
        + config.loss.pinball_weight * pinball
        + crossing_weight * crossing
    )
    return total, mse, pinball, crossing


def sample_quantiles(batch_size, config, device):
    return torch.empty(batch_size, device=device).uniform_(config.quantile.minimum, config.quantile.maximum)


def crop_prediction(prediction, target, config):
    if config.modes.full_fov_loss:
        return prediction
    prediction, _ = crop_borders((prediction, target), config, batch=True)
    return prediction


def run_epoch(model, loader, optimizer, config, device, epoch, max_steps=None):
    training = optimizer is not None
    model.train(training)
    total = {"loss": 0.0, "mse": 0.0, "pinball": 0.0, "crossing": 0.0, "count": 0}
    iterator = tqdm(loader, desc=("train" if training else "validation") + f" {epoch + 1}")
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, (measurement, target, _) in enumerate(iterator):
            if max_steps is not None and step >= max_steps:
                break
            measurement, target = measurement.to(device, non_blocking=True), target.to(device, non_blocking=True)
            quantile = sample_quantiles(measurement.shape[0], config, device)
            crossing_weight = float(OmegaConf.select(config, "loss.crossing_weight", default=0.0))
            endpoints = OmegaConf.select(config, "quantile.train_endpoints", default=False)
            if endpoints:
                low_q = torch.full_like(quantile, float(config.quantile.minimum))
                high_q = torch.full_like(quantile, float(config.quantile.maximum))
                reconstruction, predictions = model.forward_quantiles(measurement, [low_q, high_q])
                low_prediction, high_prediction = [crop_prediction(p, target, config) for p in predictions]
                crossing = F.relu(low_prediction - high_prediction).mean()
                quantile_reconstruction = predictions[0]
                quantile = low_q
            elif training and crossing_weight > 0.0:
                low_quantile = torch.empty_like(quantile).uniform_(config.quantile.minimum, 0.5)
                high_quantile = torch.empty_like(quantile).uniform_(0.5, config.quantile.maximum)
                reconstruction, predictions = model.forward_quantiles(
                    measurement, [quantile, low_quantile, high_quantile]
                )
                quantile_reconstruction, low_prediction, high_prediction = predictions
                crossing = F.relu(
                    crop_prediction(low_prediction, target, config)
                    - crop_prediction(high_prediction, target, config)
                ).mean()
            else:
                reconstruction, quantile_reconstruction = model(measurement, quantile)
                crossing = None
            reconstruction, quantile_reconstruction, target = crop_outputs(
                reconstruction, quantile_reconstruction, target, config
            )
            loss, mse, pinball, crossing = combined_loss(
                reconstruction, quantile_reconstruction, target, quantile, config, crossing
            )
            if endpoints:
                endpoint_pinball = 0.5 * (
                    pinball_loss(low_prediction, target, low_q)
                    + pinball_loss(high_prediction, target, high_q)
                )
                loss = loss + config.loss.pinball_weight * (endpoint_pinball - pinball)
                pinball = endpoint_pinball
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimization.max_grad_norm)
                optimizer.step()
            total["loss"] += loss.item()
            total["mse"] += mse.item()
            total["pinball"] += pinball.item()
            total["crossing"] += crossing.item()
            total["count"] += 1
            iterator.set_postfix(loss=f"{loss.item():.4g}", mse=f"{mse.item():.4g}", pinball=f"{pinball.item():.4g}")
    if total["count"] == 0:
        raise RuntimeError("No batches were processed")
    return {key: value / total["count"] for key, value in total.items() if key != "count"}


@torch.no_grad()
def evaluate(model, loader, config, device, max_steps=None):
    model.eval()
    psnr = PSNR(data_range=1.0).to(device)
    ssim = SSIM(data_range=1.0).to(device)
    pinball_by_q = {str(q): 0.0 for q in config.quantile.validation_levels}
    coverage_by_q = {str(q): 0.0 for q in config.quantile.validation_levels}
    q_crossing_total = 0.0
    count = 0
    for step, (measurement, target, _) in enumerate(tqdm(loader, desc="metrics")):
        if max_steps is not None and step >= max_steps:
            break
        measurement, target = measurement.to(device, non_blocking=True), target.to(device, non_blocking=True)
        full_target = target
        batch_size = measurement.shape[0]
        median_q = torch.full((batch_size,), 0.5, device=device)
        validation_quantiles = [
            torch.full((batch_size,), float(q_value), device=device)
            for q_value in config.quantile.validation_levels
        ]
        reconstruction, all_predictions = model.forward_quantiles(
            measurement, [median_q, *validation_quantiles]
        )
        median_prediction, *validation_predictions = all_predictions
        reconstruction, median_prediction, cropped_target = crop_outputs(reconstruction, median_prediction, full_target, config)
        psnr.update(reconstruction, cropped_target)
        ssim.update(reconstruction, cropped_target)
        for q_value, q, prediction in zip(
            config.quantile.validation_levels, validation_quantiles, validation_predictions
        ):
            if config.modes.full_fov_loss:
                quantile_target = full_target
            else:
                prediction, quantile_target = crop_borders((prediction, full_target), config, batch=True)
            pinball_by_q[str(q_value)] += pinball_loss(prediction, quantile_target, q).item()
            coverage_by_q[str(q_value)] += (quantile_target <= prediction).float().mean().item()
            if float(q_value) == min(config.quantile.validation_levels):
                lower_prediction = prediction
            if float(q_value) == max(config.quantile.validation_levels):
                upper_prediction = prediction
        if lower_prediction is not None and upper_prediction is not None:
            q_crossing_total += (lower_prediction > upper_prediction).float().mean().item()
        count += 1
    if count == 0:
        raise RuntimeError("No validation batches were processed")
    return {
        "reconstruction_psnr": float(psnr.compute()),
        "reconstruction_ssim": float(ssim.compute()),
        "quantile_pinball": {key: value / count for key, value in pinball_by_q.items()},
        "quantile_coverage": {key: value / count for key, value in coverage_by_q.items()},
        "q_crossing_fraction": q_crossing_total / count,
    }


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    seed = int(OmegaConf.select(config, "run.seed", default=20260920))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_setup.gpu_visible_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(config.gpu_setup.gpu_num) if device.type == "cuda" else None

    train_loader, val_loader, _ = get_loader(config)
    model = QuantileConvRecon(
        n_channels=config.model.n_channels,
        height=config.model.output_height,
        width=config.model.output_width,
        model_size=config.model.size,
        drop_path_rate=config.model.drop_path_rate,
        layer_scale_init_value=config.model.layer_scale_init_value,
        decoder_channels=config.model.decoder_channels,
        quantile_normalization=OmegaConf.select(config, "quantile.normalization", default="batch"),
        quantile_embedding_dim=config.quantile.embedding_dim,
        quantile_condition_channels=config.quantile.condition_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimization.lr, weight_decay=config.optimization.weight_decay)
    run_name = config.run.name
    output_dir = Path(config.checkpoint.save_checkpoint_path) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_mse = float("inf")
    history = []
    start_epoch = 0
    resume_path = OmegaConf.select(config, "resume.checkpoint_path", default="")
    initialization_path = OmegaConf.select(config, "initialization.model_path", default="")
    if resume_path:
        resume_checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_mse = float(resume_checkpoint.get("best_mse", resume_checkpoint["metrics"]["validation"]["mse"]))
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open() as handle:
                history = json.load(handle)
        # Preserve the starting best model when continuing into a new run directory.
        if not (output_dir / "best_model.pth").exists():
            baseline_checkpoint = dict(resume_checkpoint)
            baseline_checkpoint["config"] = OmegaConf.to_container(config, resolve=True)
            torch.save(baseline_checkpoint, output_dir / "best_model.pth")
        print(f"Resumed epoch {start_epoch + 1} from {resume_path}")
    elif initialization_path:
        initial_checkpoint = torch.load(initialization_path, map_location=device)
        model.load_state_dict(initial_checkpoint["model_state_dict"])
        print(f"Initialized model weights from {initialization_path}")
    print(f"device={device}; parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)}")

    started_at = time.monotonic()
    epoch_durations = []
    budget_seconds = 60 * float(OmegaConf.select(config, "run.max_minutes", default=0))
    for epoch in range(start_epoch, config.model.num_epochs):
        if budget_seconds and epoch_durations and time.monotonic() - started_at + max(epoch_durations) * 1.15 > budget_seconds:
            print("Stopping before next epoch to preserve checkpoint within time budget", flush=True)
            break
        epoch_started_at = time.monotonic()
        train_stats = run_epoch(model, train_loader, optimizer, config, device, epoch, args.max_train_steps)
        val_stats = run_epoch(model, val_loader, None, config, device, epoch, args.max_val_steps)
        metrics = evaluate(model, val_loader, config, device, args.max_val_steps)
        record = {"epoch": epoch + 1, "train": train_stats, "validation": val_stats, "metrics": metrics}
        history.append(record)
        print(json.dumps(record, indent=2))
        improved = val_stats["mse"] < best_mse
        best_mse = min(best_mse, val_stats["mse"])
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": OmegaConf.to_container(config, resolve=True), "metrics": record, "best_mse": best_mse}
        torch.save(checkpoint, output_dir / "latest_model.pth")
        if improved:
            best_mse = val_stats["mse"]
            torch.save(checkpoint, output_dir / "best_model.pth")
        with (output_dir / "metrics.json").open("w") as handle:
            json.dump(history, handle, indent=2)
        epoch_durations.append(time.monotonic() - epoch_started_at)


if __name__ == "__main__":
    main()
