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
from models import make_model
from ConvNeXt import ConvRecon
from models.ensemble import DeepEnsemble, DeepEnsembleMember

EPOCHS = 50


def learning_rate(step, total_steps):
    """Nine-epoch linear warmup, then cosine decay to 1e-9 at epoch 50."""
    progress = step / (total_steps - 1)
    warmup = 9 / EPOCHS
    if progress <= warmup:
        return 1e-5 + (1e-4 - 1e-5) * progress / warmup
    # Personal choice
    cosine = (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup))) / 2
    return 1e-9 + (1e-4 - 1e-9) * cosine


def train_epoch(
    model, loader, optimizer, device, global_step, total_steps,
    batch_order=None, loss_divisor=None,
):
    """Train each ensemble member with its own order of this epoch's batches."""
    model.train()
    members = model.members if isinstance(model, DeepEnsemble) else [model]
    if batch_order is None:
        batches = list(loader.batch_sampler)
        orders = [batches]
        if isinstance(model, DeepEnsemble):
            orders = [
                [batches[i] for i in torch.randperm(len(batches)).tolist()]
                for _ in members
            ]
    else:
        orders = [batch_order]  # One independently scheduled ensemble member.

    if loss_divisor is None:
        loss_divisor = len(members)
    loss_sum = 0.0
    images_seen = 0
    for member, order in zip(members, orders):
        member_loader = DataLoader(
            loader.dataset, batch_sampler=order,
            num_workers=loader.num_workers, collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            # Worker setup must not consume the model's restored RNG stream.
            generator=torch.Generator().manual_seed(777),
        )
        for batch_index, (measurement, target) in enumerate(member_loader):
            measurement, target = measurement.to(device), target.to(device)
            step = global_step + batch_index
            schedule_progress = step / (total_steps - 1)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total_steps)
                group["weight_decay"] = 1e-4 * (1 + math.cos(math.pi * schedule_progress)) / 2
            # Inactive members must have no gradients so AdamW skips them.
            optimizer.zero_grad()
            loss = member.loss(measurement, target)
            (loss / loss_divisor).backward()
            optimizer.step()
            loss_sum += loss.item() * len(measurement)
            images_seen += len(measurement)
    return loss_sum / images_seen, global_step + len(loader)


def save_checkpoint(checkpoint, path):
    """Replace a recovery file only after the complete new file is written."""
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def rng_state():
    """Capture the random streams used by quantiles and dropout."""
    return dict(
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    """Restore streams after reconstructing the model and optimizer."""
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


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
        lpips_sum += float(perceptual(prediction.clip(0, 1), target).sum())
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
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--ensemble-member", type=int,
                        help="Train this ensemble member (0–9) in its own GPU job.")
    args = parser.parse_args()
    torch.manual_seed(777 + (args.ensemble_member or 0))  # personal preference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = get_loaders(args.manifest)
    _, sample_target = train_loader.dataset[0]
    height, width = sample_target.shape[-2:]
    if args.ensemble_member is None:
        model = make_model(args.model, height, width, len(train_loader.dataset))
    else:
        model = DeepEnsemble([DeepEnsembleMember(
            ConvRecon(3, height, width, out_channels=6)
        )])
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    perceptual = None if args.ensemble_member is not None else (
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
    completed_epoch = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("train_progress"):
            raise ValueError("Finish this partial epoch with the previous trainer before resuming.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, global_step = saved["history"], saved["global_step"]
        completed_epoch = saved["epoch"]
        best_ssim, best_checkpoint = saved["best_ssim"], saved["best_checkpoint"]
        restore_rng(saved["rng"])
        del saved

    def checkpoint_state(completed):
        return dict(
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=completed, method=args.model, history=history,
            height=height, width=width, dataset_size=len(train_loader.dataset),
            global_step=global_step,
            best_ssim=best_ssim, best_checkpoint=best_checkpoint, rng=rng_state(),
            ensemble_member=args.ensemble_member,
        )

    for epoch in range(completed_epoch + 1, EPOCHS + 1):
        batch_order = None
        if args.ensemble_member is not None:
            from ensemble_jobs import member_batch_order
            batch_order = member_batch_order(train_loader, epoch, args.ensemble_member)
        loss, global_step = train_epoch(
            model, train_loader, optimizer, device, global_step, total_steps,
            batch_order=batch_order,
            loss_divisor=10 if args.ensemble_member is not None else None,
        )
        # Ensemble validation uses all ten members together in a separate job.
        scores = (validate(model, validation_loader, perceptual, device)
                  if args.ensemble_member is None else {})
        # Writing result to json
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

        if args.ensemble_member is not None:
            save_checkpoint(dict(
                model=model.state_dict(), epoch=epoch, method="ensemble",
                ensemble_member=args.ensemble_member, height=height, width=width,
                dataset_size=len(train_loader.dataset),
            ), args.output / f"epoch_{epoch:03d}.pt")
            save_checkpoint(checkpoint_state(epoch), args.output / "latest.pt")
            continue

        # Checkpoint
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
        save_checkpoint(checkpoint_state(epoch), args.output / "latest.pt")


if __name__ == "__main__":
    main()
