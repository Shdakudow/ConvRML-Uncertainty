"""Fresh 0-50 epoch joint adaptive-q training; no pretrained checkpoint loading."""

import argparse, json, math, signal, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader, Subset
from models.corrected_uncertainty import PinballQUTCC
from models.constrained_adaptive_qutcc import (
    ConstrainedAdaptiveQUTCC,
    joint_step,
    PROTOCOL,
)
from scripts.corrected_comparison_study import paths
from scripts.formal_rml_train import Pairs
from scripts.training_records import manifest_digest
from scripts.train_image_adaptive_qutcc import validate

ROOT = Path("results/adaptive_q_coverage90_scratch50_20261014")
STOP = False


def stop(*_):
    global STOP
    STOP = True


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def save_torch(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def rank(v):
    violation = max(0.0, 0.9 - v["coverage"])
    return (violation > 0, violation, v["width"])


def load_checkpoint(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["protocol"] != PROTOCOL or ck["initialization"] != "random_no_pretrained":
        raise ValueError("Wrong protocol")
    m = ConstrainedAdaptiveQUTCC(PinballQUTCC())
    m.load_state_dict(ck["model"], strict=True)
    return m.to(device).eval(), ck


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["smoke", "train"])
    p.add_argument("--camera", choices=["rml", "diffuser"], required=True)
    p.add_argument("--target-epoch", type=int, default=2)
    p.add_argument("--save-steps", type=int, default=1000)
    a = p.parse_args()
    if not 1 <= a.target_epoch <= 50 or a.save_steps < 1:
        raise ValueError("Invalid training length")
    torch.set_num_threads(2)
    seed = 202610140 + (a.camera == "diffuser")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out = ROOT / a.camera
    out.mkdir(parents=True, exist_ok=True)
    train = Pairs("train", paths(a.camera)[0])
    val = Pairs("val", paths(a.camera)[0])
    if len(train) != 60000 or len(val) != 5000:
        raise ValueError("Wrong split")
    # This constructor creates random weights; no load_model()/source checkpoint.
    model = ConstrainedAdaptiveQUTCC(PinballQUTCC()).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    qrng = torch.Generator(device="cuda").manual_seed(seed + 1000)
    meta = dict(
        protocol=PROTOCOL,
        camera=a.camera,
        initialization="random_no_pretrained",
        source_epoch=0,
        epochs=50,
        epochs_per_job=2,
        batch_size=2,
        lr=1e-4,
        weight_decay=1e-4,
        seed=seed,
        train_count=len(train),
        val_count=len(val),
        test_count=5000,
        calibration_count=2048,
        manifest_sha256=manifest_digest(paths(a.camera)[0]),
        alpha=0.1,
        delta=0.05,
        width_weight=model.width_weight,
        tau=model.tau,
        dual_rate=model.controller.rate,
        dual_momentum=model.controller.momentum,
        training_q_mixture=dict(uniform=0.5, learned_lower=0.25, learned_upper=0.25),
        objective="pinball + 0.1*(width + adaptive_multiplier*soft_miss); multiplier updated from hard training miss rate",
        coverage_unit="mean RGB pixel/channel coverage across images; not per-image guarantee",
        precision="BF16 backbone, float32 q head",
        calibration_used=False,
        test_used=False,
    )
    if a.action == "smoke":
        small = DataLoader(Subset(val, list(range(16))), batch_size=2)
        before = validate(model, small)
        base_before = next(model.quantile_model.parameters()).detach().clone()
        head_before = model.selector.head.bias.detach().clone()
        samples = []
        times = []
        for b in DataLoader(Subset(train, list(range(32))), batch_size=2):
            torch.cuda.synchronize()
            t = time.monotonic()
            r = joint_step(model, b[:, 0].cuda(), b[:, 1].cuda(), opt, qrng)
            torch.cuda.synchronize()
            times.append(time.monotonic() - t)
            samples.append(r)
        t = time.monotonic()
        after = validate(model, small)
        torch.cuda.synchronize()
        val_seconds = time.monotonic() - t
        estimate = (
            sum(times[4:]) / len(times[4:]) * 30000 + val_seconds * 5000 / 16
        ) * 1.3
        base_changed = not torch.equal(
            base_before, next(model.quantile_model.parameters())
        )
        head_changed = not torch.equal(head_before, model.selector.head.bias)
        passed = (
            base_changed
            and head_changed
            and max(r["selector_grad"] for r in samples) > 0
            and estimate < 10.5 * 3600
        )
        report = dict(
            **meta,
            passed=passed,
            backbone_changed=base_changed,
            head_changed=head_changed,
            estimated_epoch_hours=estimate / 3600,
            gpu=torch.cuda.get_device_name(),
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
            before=before,
            after=after,
            steps=samples,
        )
        write(out / "smoke.json", report)
        print(json.dumps(report), flush=True)
        if not passed:
            raise RuntimeError("Fresh-model smoke gate failed")
        return
    smoke = json.loads((out / "smoke.json").read_text())
    if not smoke["passed"] or any(
        smoke[k] != meta[k]
        for k in ["protocol", "initialization", "seed", "manifest_sha256"]
    ):
        raise ValueError("Smoke gate mismatch")
    latest = out / "latest.pt"
    epoch, offset, history = 1, 0, []
    best = None
    running = {"pinball": 0.0, "width": 0.0, "hard_risk": 0.0, "batches": 0}
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        for key in [
            "protocol",
            "initialization",
            "seed",
            "manifest_sha256",
            "width_weight",
            "tau",
            "lr",
            "dual_rate",
            "dual_momentum",
        ]:
            if ck[key] != meta[key]:
                raise ValueError("Resume mismatch: " + key)
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["cpu_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        qrng.set_state(ck["q_rng"])
        epoch, offset, history, best, running = (
            ck["next_epoch"],
            ck["next_offset"],
            ck["history"],
            ck["best_rank"],
            ck["running"],
        )
        del ck

    def save(next_epoch, next_offset):
        save_torch(
            latest,
            dict(
                **meta,
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                q_rng=qrng.get_state(),
                next_epoch=next_epoch,
                next_offset=next_offset,
                history=history,
                best_rank=best,
                running=running,
            ),
        )

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    write(out / "config.json", meta)
    if not latest.exists():
        save(1, 0)
        save_torch(
            out / "epoch_000.pt", dict(**meta, epoch=0, model=model.state_dict())
        )
    while epoch <= a.target_epoch:
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
        loader = DataLoader(
            Subset(train, order[offset:]),
            batch_size=2,
            num_workers=2,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed + epoch + 10000),
        )
        start = time.monotonic()
        for b in loader:
            r = joint_step(
                model,
                b[:, 0].cuda(non_blocking=True),
                b[:, 1].cuda(non_blocking=True),
                opt,
                qrng,
            )
            offset += len(b)
            running["batches"] += 1
            for k in ["pinball", "width", "hard_risk"]:
                running[k] += r[k]
            if running["batches"] % 100 == 0:
                progress = dict(epoch=epoch, processed=offset, total=60000, **r)
                write(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if running["batches"] % a.save_steps == 0 or STOP:
                save(epoch, offset)
            if STOP:
                raise SystemExit(75)
        metrics = validate(model, DataLoader(val, batch_size=2, num_workers=2))
        row = dict(
            epoch=epoch,
            validation=metrics,
            multiplier=float(model.controller.multiplier),
            risk_ema=float(model.controller.ema_risk),
            segment_seconds=time.monotonic() - start,
            gpu=torch.cuda.get_device_name(),
            training={
                k: running[k] / running["batches"]
                for k in ["pinball", "width", "hard_risk"]
            },
        )
        history.append(row)
        candidate = rank(metrics)
        compact = dict(**meta, epoch=epoch, model=model.state_dict(), history=history)
        if epoch % 5 == 0:
            save_torch(out / ("epoch_%03d.pt" % epoch), compact)
        if best is None or candidate < tuple(best):
            best = candidate
            save_torch(out / "best.pt", compact)
            write(
                out / "best.json",
                dict(
                    epoch=epoch,
                    coverage=metrics["coverage"],
                    width=metrics["width"],
                    feasible=metrics["coverage"] >= 0.9,
                ),
            )
        write(out / "history.json", history)
        print(json.dumps(row), flush=True)
        epoch += 1
        offset = 0
        running = {"pinball": 0.0, "width": 0.0, "hard_risk": 0.0, "batches": 0}
        save(epoch, offset)


if __name__ == "__main__":
    main()
