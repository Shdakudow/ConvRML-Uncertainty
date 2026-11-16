"""Split sequential ensemble recovery files and evaluate parallel members."""

import argparse
import copy
import json
from pathlib import Path

import torch


def member_batch_progress(loader, epoch, member_index):
    """Share batch membership, with an independent batch order for each member."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(777 + epoch).get_state())
        batches = list(loader.batch_sampler)
    generator = torch.Generator().manual_seed(777 + 10 * epoch + member_index)
    order = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
    return dict(orders=[order], member=0, batch=0, loss_sum=0.0, count=0)


def split_member_checkpoint(saved, member_index):
    """Keep one member's weights, AdamW moments, schedule, and pending batches."""
    prefix = f"members.{member_index}."
    model = {"members.0." + name[len(prefix):]: value
             for name, value in saved["model"].items() if name.startswith(prefix)}
    optimizer = dict(state={}, param_groups=copy.deepcopy(saved["optimizer"]["param_groups"]))
    # The shared AdamW group lists ten equally sized member parameter lists.
    group = optimizer["param_groups"][0]
    width = len(group["params"]) // 10
    parameter_ids = group["params"][member_index * width:(member_index + 1) * width]
    optimizer["state"] = {i: saved["optimizer"]["state"][old]
                          for i, old in enumerate(parameter_ids) if old in saved["optimizer"]["state"]}
    group["params"] = list(range(width))
    progress = saved["train_progress"]
    completed_epoch = saved["epoch"]
    step = saved["global_step"]
    pending = None
    if progress:
        batches = progress["orders"][member_index]
        if member_index < progress["member"]:
            completed_epoch += 1
            step += len(batches)
        else:
            cursor = progress["batch"] if member_index == progress["member"] else 0
            pending = dict(orders=[batches], member=0, batch=cursor, loss_sum=0.0, count=0)
    return dict(
        model=model, optimizer=optimizer, epoch=completed_epoch, method="ensemble",
        ensemble_member=member_index, height=saved["height"], width=saved["width"],
        dataset_size=saved["dataset_size"], global_step=step, train_progress=pending,
        history=[], best_ssim=-float("inf"), best_checkpoint=None,
        rng=dict(cpu=torch.Generator().manual_seed(777 + member_index).get_state(), cuda=[]),
        migration=dict(source_completed_epoch=saved["epoch"],
                       resumed_epoch_loss_excludes_batches_before_split=True),
    )


def split_checkpoint(source, output):
    from train import save_checkpoint

    saved = torch.load(source, map_location="cpu", weights_only=False)
    output.mkdir(parents=True, exist_ok=True)
    if saved["best_checkpoint"] is not None:
        save_checkpoint(saved["best_checkpoint"], output / "best.pt")
    (output / "history.json").write_text(json.dumps(saved["history"], indent=2) + "\n")
    report = dict(source=str(source), completed_epoch=saved["epoch"], members=[])
    for index in range(10):
        member = split_member_checkpoint(saved, index)
        folder = output / f"member_{index:02d}"
        folder.mkdir(exist_ok=True)
        save_checkpoint(member, folder / "latest.pt")
        if member["epoch"] > saved["epoch"]:
            weights = {key: member[key] for key in
                       ("model", "epoch", "method", "ensemble_member", "height", "width", "dataset_size")}
            save_checkpoint(weights, folder / f"epoch_{member['epoch']:03d}.pt")
        report["members"].append(dict(index=index, completed_epoch=member["epoch"],
                                      pending_batch=(member["train_progress"] or {}).get("batch")))
    (output / "migration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def ready_epochs(output):
    """Return consecutive epochs whose ten member snapshots are available."""
    history = json.loads((output / "history.json").read_text())
    epoch = max((row["epoch"] for row in history), default=0) + 1
    epochs = []
    while epoch <= 50 and all(
        (output / f"member_{index:02d}" / f"epoch_{epoch:03d}.pt").exists()
        for index in range(10)
    ):
        epochs.append(epoch)
        epoch += 1
    return epochs


def evaluate_ready(manifest, output):
    from dataloader import get_loaders
    from models import make_model
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from train import save_checkpoint, validate

    _, loader = get_loaders(manifest)
    best_file = output / "best.pt"
    best_ssim = (torch.load(best_file, map_location="cpu", weights_only=False)["validation_ssim"]
                 if best_file.exists() else -float("inf"))
    for epoch in ready_epochs(output):
        first = torch.load(output / "member_00" / f"epoch_{epoch:03d}.pt",
                           map_location="cpu", weights_only=False)
        model = make_model("ensemble", first["height"], first["width"], first["dataset_size"])
        for index, member in enumerate(model.members):
            state = (first if index == 0 else torch.load(
                output / f"member_{index:02d}" / f"epoch_{epoch:03d}.pt",
                map_location="cpu", weights_only=False))
            member.load_state_dict({key.removeprefix("members.0."): value
                                    for key, value in state["model"].items()})
        model = model.cuda()
        perceptual = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True, reduction="none"
        ).cuda().eval().requires_grad_(False)
        scores = validate(model, loader, perceptual, "cuda")
        weights = dict(model={name: value.detach().cpu() for name, value in model.state_dict().items()},
                       epoch=epoch, method="ensemble", validation_ssim=scores["ssim"],
                       height=first["height"], width=first["width"], dataset_size=first["dataset_size"])
        save_checkpoint(weights, output / "latest.pt")
        if scores["ssim"] > best_ssim:
            best_ssim = scores["ssim"]
            save_checkpoint(weights, best_file)
        if epoch % 5 == 0:
            best = torch.load(best_file, map_location="cpu", weights_only=False)
            save_checkpoint(dict(best, selected_through_epoch=epoch),
                            output / f"best_through_epoch_{epoch:03d}.pt")
        history_file = output / "history.json"
        history = json.loads(history_file.read_text())
        row = dict(epoch=epoch, **scores)
        history.append(row)
        temporary = history_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(history, indent=2) + "\n")
        temporary.replace(history_file)
        print(json.dumps(row), flush=True)
        del model, perceptual, weights, state, first
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--source", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "split":
        split_checkpoint(args.source, args.output)
    else:
        evaluate_ready(args.manifest, args.output)


if __name__ == "__main__":
    main()
