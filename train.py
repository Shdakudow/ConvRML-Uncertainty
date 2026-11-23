"""Train one uncertainty method for 50 epochs on paired RGB images."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dataloader import get_loaders
from models import make_model
from models.common import INITIAL_WEIGHT_DECAY
from ConvNeXt import ConvRecon
from models.ensemble import DeepEnsemble, DeepEnsembleMember

EPOCHS = 50


def learning_rate(step, total_steps):
    """Nine-epoch linear warmup, then cosine decay to 1e-9 at epoch 50."""
    progress = step / (total_steps - 1)
    warmup = 9 / EPOCHS
    if progress <= warmup:
        return 1e-5 + (1e-4 - 1e-5) * progress / warmup
    cosine = (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup))) / 2
    return 1e-9 + (1e-4 - 1e-9) * cosine


def train_epoch(member, loader, optimizer, device, global_step, total_steps, loss_divisor=1):
    member.train()
    loss_sum = 0.0
    images_seen = 0
    for batch_index, (measurement, target) in enumerate(loader):
        measurement, target = measurement.to(device), target.to(device)
        step = global_step + batch_index
        schedule_progress = step / (total_steps - 1)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, total_steps)
            # Weight decay also follows a cosine schedule, decaying to 0 by the final step.
            group["weight_decay"] = INITIAL_WEIGHT_DECAY * (1 + math.cos(math.pi * schedule_progress)) / 2
        optimizer.zero_grad()
        loss = member.loss(measurement, target)
        (loss / loss_divisor).backward()
        optimizer.step()
        loss_sum += loss.item() * len(measurement)
        images_seen += len(measurement)
    return loss_sum / images_seen


def save_checkpoint(checkpoint, path):
    """Replace a recovery file only after the complete new file is written."""
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def rng_state():
    return dict(
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _make_member_loader(train_loader, batch_order):
    return DataLoader(
        train_loader.dataset, batch_sampler=batch_order,
        num_workers=train_loader.num_workers, collate_fn=train_loader.collate_fn,
        pin_memory=train_loader.pin_memory,
        # Worker setup must not consume the model's restored RNG stream.
        generator=torch.Generator().manual_seed(777),
    )


@torch.inference_mode()
def validate(model, loader, perceptual, device):
    model.eval()
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
        lpips_sum += float(perceptual(prediction.clip(0, 1), target).sum())
        perceptual.reset()
        count += len(measurement)
    return dict(
        psnr=float(psnr.compute()), ssim=float(ssim.compute()), lpips=lpips_sum / count
    )


def train(model, train_loader, validation_loader, optimizer, device,
          total_steps, completed_epoch, history, global_step,
          best_ssim, best_checkpoint, args):
    _, sample_target = train_loader.dataset[0]
    height, width = sample_target.shape[-2:]

    perceptual = (
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True, reduction="none")
        .to(device).eval().requires_grad_(False)
    )

    for epoch in range(completed_epoch + 1, EPOCHS + 1):
        batches = list(train_loader.batch_sampler)
        if isinstance(model, DeepEnsemble):
            members = model.members
            orders = [
                [batches[i] for i in torch.randperm(len(batches)).tolist()]
                for _ in members
            ]
        else:
            members = [model]
            orders = [batches]
        losses = [
            train_epoch(
                member, _make_member_loader(train_loader, order),
                optimizer, device, global_step, total_steps, loss_divisor=len(members),
            )
            for member, order in zip(members, orders)
        ]
        loss = sum(losses) / len(losses)
        global_step += len(train_loader)

        scores = validate(model, validation_loader, perceptual, device)
        row = dict(epoch=epoch, loss=loss, **scores,
                   learning_rate=optimizer.param_groups[0]["lr"],
                   weight_decay=optimizer.param_groups[0]["weight_decay"])
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")

        if scores["ssim"] > best_ssim:
            best_ssim = scores["ssim"]
            best_checkpoint = dict(
                # Clone CPU tensors so later optimizer updates cannot change
                # the saved best weights, including BatchNorm buffers.
                model={
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                epoch=epoch, method=args.model, validation_ssim=best_ssim,
                height=height, width=width, dataset_size=len(train_loader.dataset),
            )
        # Snapshot the best model so far every 5 epochs for mid-run analysis.
        if epoch % 5 == 0:
            torch.save(
                dict(best_checkpoint, selected_through_epoch=epoch),
                args.output / f"best_through_epoch_{epoch:03d}.pt",
            )
        save_checkpoint(dict(
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch, method=args.model, history=history,
            height=height, width=width, dataset_size=len(train_loader.dataset),
            global_step=global_step, best_ssim=best_ssim, best_checkpoint=best_checkpoint,
            rng=rng_state(),
        ), args.output / "latest.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["qutcc", "im2im", "ensemble", "mcdropout"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    # Distributed ensemble: torchrun sets RANK automatically; each rank trains one member.
    # Usage: torchrun --nproc_per_node=10 train.py --model ensemble ...
    distributed = args.model == "ensemble" and "RANK" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        args.output = args.output / f"member_{rank:02d}"
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(777 + rank)
    train_loader, validation_loader = get_loaders(args.manifest)
    _, sample_target = train_loader.dataset[0]
    height, width = sample_target.shape[-2:]

    if distributed:
        model = DeepEnsemble([DeepEnsembleMember(ConvRecon(3, height, width, out_channels=6))])
    else:
        model = make_model(args.model, height, width, len(train_loader.dataset))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=INITIAL_WEIGHT_DECAY)

    total_steps = EPOCHS * len(train_loader)
    args.output.mkdir(parents=True, exist_ok=True)
    history, global_step = [], 0
    best_ssim = -float("inf")
    best_checkpoint = None
    completed_epoch = 0

    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, global_step = saved["history"], saved["global_step"]
        completed_epoch = saved["epoch"]
        best_ssim, best_checkpoint = saved["best_ssim"], saved["best_checkpoint"]
        restore_rng(saved["rng"])
        del saved

    train(model, train_loader, validation_loader, optimizer, device,
          total_steps, completed_epoch, history, global_step,
          best_ssim, best_checkpoint, args)


if __name__ == "__main__":
    main()
