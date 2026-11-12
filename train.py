"""Train one uncertainty method for 50 epochs on paired RGB images."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dataloader import get_loaders
from Models import make_model
from Models.ensemble import DeepEnsemble

EPOCHS = 50


def learning_rate(step, total_steps):
    """Nine-epoch linear warmup, then cosine decay to 1e-9 at epoch 50."""
    progress = step / (total_steps - 1)
    warmup = 9 / EPOCHS
    if progress <= warmup:
        return 1e-5 + (1e-4 - 1e-5) * progress / warmup
    cosine = (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup))) / 2
    return 1e-9 + (1e-4 - 1e-9) * cosine


def train_epoch(model, loader, optimizer, device, global_step, total_steps):
    """Train each ensemble member with its own order of this epoch's batches."""
    model.train()
    if isinstance(model, DeepEnsemble):
        members = model.members
        # Form batches once, then permute their order separately for each member.
        # Store indices only, rather than keeping the dataset's images in RAM.
        batches = list(loader.batch_sampler)
        member_loaders = [
            DataLoader(
                loader.dataset,
                batch_sampler=[
                    batches[index]
                    for index in torch.randperm(len(batches)).tolist()
                ],
                num_workers=loader.num_workers,
                collate_fn=loader.collate_fn,
                pin_memory=loader.pin_memory,
            )
            for _ in members
        ]
    else:
        members, member_loaders = [model], [loader]

    loss_sum = 0.0
    count = 0
    for member, member_loader in zip(members, member_loaders):
        # Each member starts at the same point in the epoch-based schedule.
        step = global_step
        for measurement, target in member_loader:
            measurement, target = measurement.to(device), target.to(device)
            progress = step / (total_steps - 1)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total_steps)
                group["weight_decay"] = 1e-4 * (1 + math.cos(math.pi * progress)) / 2
            # Inactive members must have no gradients so AdamW skips them.
            optimizer.zero_grad(set_to_none=True)
            loss = member.loss(measurement, target)
            (loss / len(members)).backward()
            optimizer.step()
            loss_sum += loss.item() * len(measurement)
            count += len(measurement)
            step += 1
    return loss_sum / count, step


@torch.inference_mode()
def validate(model, loader, perceptual, device):
    """Measure reconstruction only; calibration is a separate held-out step."""
    model.eval()
    # Fresh metric objects keep each validation epoch's statistics separate.
    # Reducing over C/H/W preserves mean per-image PSNR rather than global PSNR.
    psnr = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3)).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_sum = 0.0
    count = 0
    for measurement, target in loader:
        measurement, target = measurement.to(device), target.to(device)
        prediction = model.center(measurement)
        psnr.update(prediction, target)
        ssim.update(prediction, target)
        # Evaluate LPIPS on a clipped copy; PSNR and SSIM use raw predictions.
        lpips_sum += float(perceptual(prediction.clamp(0, 1), target).sum())
        perceptual.reset()
        count += len(measurement)
    return dict(
        psnr=float(psnr.compute()), ssim=float(ssim.compute()), lpips=lpips_sum / count
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["qutcc", "im2im", "ensemble", "mcdropout"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(777)  # personal preference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = get_loaders(args.manifest)
    _, sample_target = train_loader.dataset[0]
    height, width = sample_target.shape[-2:]
    model = make_model(args.model, height, width, len(train_loader.dataset)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    perceptual = (
        LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True, reduction="none"
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    total_steps = EPOCHS * len(train_loader)
    args.output.mkdir(parents=True, exist_ok=True)
    history, global_step = [], 0
    best_ssim = -float("inf")
    best_checkpoint = None

    for epoch in range(1, EPOCHS + 1):
        loss, global_step = train_epoch(
            model, train_loader, optimizer, device, global_step, total_steps
        )
        scores = validate(model, validation_loader, perceptual, device)
        row = dict(
            epoch=epoch,
            loss=loss,
            **scores,
            learning_rate=optimizer.param_groups[0]["lr"],
            weight_decay=optimizer.param_groups[0]["weight_decay"],
        )
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            method=args.model,
            history=history,
            height=height,
            width=width,
            dataset_size=len(train_loader.dataset),
        )
        torch.save(checkpoint, args.output / "latest.pt")
        if scores["ssim"] > best_ssim:
            best_ssim = scores["ssim"]
            best_checkpoint = dict(
                # Clone CPU tensors so later optimizer updates cannot change
                # the saved best weights, including BatchNorm buffers.
                model={
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                epoch=epoch,
                method=args.model,
                validation_ssim=best_ssim,
                height=height,
                width=width,
                dataset_size=len(train_loader.dataset),
            )
        if epoch % 5 == 0:
            torch.save(
                dict(best_checkpoint, selected_through_epoch=epoch),
                args.output / f"best_through_epoch_{epoch:03d}.pt",
            )


if __name__ == "__main__":
    main()
