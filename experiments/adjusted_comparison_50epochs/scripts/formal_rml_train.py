"""Six independent workers: five full-model ensemble members and one MC model."""

import sys, json, time, os, argparse, hashlib, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from baselines.formal_rml import FormalRML
from baselines.deep_ensemble import DeepEnsembleMember
from baselines.mc_dropout import dropout_inference
from scripts.training_records import (
    manifest_digest,
    split_bounds,
    save_milestone,
    check_resume_manifest,
)

ROOT = Path("results/formal_rml_25k")
DATA = Path("data/formal_rml_25k")


class Pairs(Dataset):
    def __init__(self, role, data=None):
        data = Path(data or DATA)
        self.a = np.load(data / "pairs.npy", mmap_mode="r")
        self.start, self.count, total = split_bounds(data, role)
        if len(self.a) != total:
            raise ValueError("Cache and manifest sizes differ")

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        return (
            torch.from_numpy(np.array(self.a[self.start + i], copy=True)).float() / 255
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--worker", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3, help="0 disables early stopping")
    p.add_argument("--out", default="results/formal_rml_25k")
    p.add_argument("--resume-source", type=Path)
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--max-new-epochs", type=int, default=0)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--save-every", type=int, default=5)
    args = p.parse_args()
    global ROOT
    ROOT = Path(args.out)
    data = args.data
    manifest = manifest_digest(data)
    if args.save_every < 1:
        raise ValueError("save-every must be positive")
    if args.epochs < 1 or args.patience < 0:
        raise ValueError("Invalid epoch/patience configuration")
    if args.worker not in range(6):
        raise ValueError("worker must be 0..5")
    torch.set_num_threads(2)
    seed = 20260923 + args.worker
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / f"worker_{args.worker}"
    out.mkdir(exist_ok=True)
    method = "mc" if args.worker == 5 else "ensemble"
    net = FormalRML(method).cuda()
    model = DeepEnsembleMember(net).cuda() if method == "ensemble" else net
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train = DataLoader(
        Pairs("train", data), batch_size=2, shuffle=True, num_workers=2, pin_memory=True
    )
    val = DataLoader(Pairs("val", data), batch_size=2, num_workers=2, pin_memory=True)

    def loss(batch):
        y, x = batch[:, 0].cuda(non_blocking=True), batch[:, 1].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if method == "ensemble":
                mean, var = model(y)
                mean, var = mean.float(), var.float()
                return 0.5 * (var.log() + (mean - x).square() / var).mean()
            return (model(y).float() - x).square().mean()

    if args.smoke:
        steps = 0
        t = None
        for b in train:
            opt.zero_grad(set_to_none=True)
            v = loss(b)
            if not torch.isfinite(v):
                raise RuntimeError("Nonfinite smoke loss")
            v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            if steps == 10:
                torch.cuda.synchronize()
                t = time.monotonic()
            if steps == 40:
                break
        torch.cuda.synchronize()
        seconds = (time.monotonic() - t) / (steps - 10)
        report = {
            "method": method,
            "warmup_batches": 10,
            "timed_batches": steps - 10,
            "seconds_per_batch": seconds,
            "estimated_training_hours_8_epochs": seconds * len(train) * 8 / 3600,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "manifest_sha256": manifest,
            "passed": True,
        }
        (ROOT / f"smoke_{method}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    for name in ("ensemble", "mc"):
        smoke = json.loads(
            ((args.resume_source or ROOT) / f"smoke_{name}.json").read_text()
        )
        if (
            smoke.get("manifest_sha256", manifest) != manifest
            or not smoke["passed"]
            or smoke["seconds_per_batch"] * len(train) * 2 * 1.25 > 3 * 3600
        ):
            raise RuntimeError("Smoke runtime gate failed; review before long training")
    history = []
    best = float("inf")
    stale = 0
    start_epoch = 0
    resume_dir = (
        out
        if (out / "latest.pt").exists()
        else (
            args.resume_source / f"worker_{args.worker}" if args.resume_source else out
        )
    )
    if args.resume_source and not (resume_dir / "latest.pt").exists():
        raise FileNotFoundError(resume_dir / "latest.pt")
    if args.resume_source:
        previous = json.loads(
            (args.resume_source / f"worker_{args.worker}/complete.json").read_text()
        )
        if previous["manifest_sha256"] != manifest:
            raise ValueError("Resume split mismatch")
    if (resume_dir / "latest.pt").exists():
        ck = torch.load(
            resume_dir / "latest.pt", map_location="cpu", weights_only=False
        )
        legacy_path = resume_dir / "complete.json"
        legacy = (
            json.loads(legacy_path.read_text()).get("manifest_sha256")
            if legacy_path.exists()
            else None
        )
        check_resume_manifest(ck, manifest, legacy)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        history = ck["history"]
        best = ck["best"]
        stale = ck["stale"]
        start_epoch = ck["epoch"]
        torch.set_rng_state(ck["rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        if resume_dir != out:
            shutil.copy2(resume_dir / "best.pt", out / "best.pt")
        del ck
    if start_epoch > args.epochs:
        raise ValueError("Requested end epoch precedes checkpoint")
    if args.max_new_epochs and args.epochs - start_epoch > args.max_new_epochs:
        raise ValueError("Checkpoint is too early for this two-epoch chunk")
    (out / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(
        json.dumps(
            {
                "worker": args.worker,
                "resume_epoch": start_epoch,
                "target_epoch": args.epochs,
                "optimizer_lrs": [g["lr"] for g in opt.param_groups],
            }
        ),
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        start = time.monotonic()
        total = 0.0
        count = 0
        for b in train:
            opt.zero_grad(set_to_none=True)
            v = loss(b)
            if not torch.isfinite(v):
                raise RuntimeError("Nonfinite training loss")
            v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += v.item() * len(b)
            count += len(b)
        model.eval()
        vsum = 0.0
        with torch.inference_mode():
            for b in val:
                vsum += loss(b).item() * len(b)
        vmean = vsum / len(val.dataset)
        if not __import__("math").isfinite(vmean):
            raise RuntimeError("Nonfinite validation loss")
        if vmean < best:
            best = vmean
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation": best,
                    "manifest_sha256": manifest,
                },
                out / "best.tmp",
            )
            (out / "best.tmp").replace(out / "best.pt")
        else:
            stale += 1
        row = {
            "epoch": epoch + 1,
            "train_loss": total / count,
            "validation_loss": vmean,
            "seconds": time.monotonic() - start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best": best,
                "stale": stale,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "manifest_sha256": manifest,
                "run_arguments": vars(args),
            },
            out / "latest.tmp",
        )
        (out / "latest.tmp").replace(out / "latest.pt")
        save_milestone(
            out,
            model,
            epoch + 1,
            args.save_every,
            manifest_sha256=manifest,
            history=history,
            method=method,
            run_arguments=vars(args),
        )
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if args.patience and stale >= args.patience:
            break
    if args.skip_inference:
        print(
            json.dumps(
                {
                    "worker": args.worker,
                    "saved_through_epoch": history[-1]["epoch"],
                    "inference_deferred": True,
                }
            ),
            flush=True,
        )
        return
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    means = np.lib.format.open_memmap(
        out / "mean.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(Pairs("test", data)), 3, 256, 256),
    )
    variances = np.lib.format.open_memmap(
        out / "variance.npy", mode="w+", dtype=np.float32, shape=means.shape
    )
    position = 0
    with torch.inference_mode():
        for b in DataLoader(Pairs("test", data), batch_size=2, num_workers=2):
            y = b[:, 0].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if method == "ensemble":
                    mu, variance = model(y)
                    mu, variance = mu.float(), variance.float()
                else:
                    with dropout_inference(net):
                        mu = None
                        m2 = None
                        for i in range(30):
                            sample = net(y).float()
                            if mu is None:
                                mu = sample.clone()
                                m2 = torch.zeros_like(sample)
                            else:
                                d = sample - mu
                                mu = mu + d / (i + 1)
                                m2 = m2 + d * (sample - mu)
                    variance = m2 / 30
            n = len(b)
            means[position : position + n] = mu.cpu().numpy()
            variances[position : position + n] = variance.cpu().numpy()
            position += n
    means.flush()
    variances.flush()
    report = {
        "worker": args.worker,
        "method": method,
        "seed": seed,
        "selected_epoch": ck["epoch"],
        "trained_through_epoch": history[-1]["epoch"],
        "manifest_sha256": manifest,
        "count": position,
        "mc_samples": 30 if method == "mc" else None,
        "variance_type": (
            "dropout epistemic only" if method == "mc" else "member learned aleatoric"
        ),
        "calibration": False,
    }
    (out / "complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
