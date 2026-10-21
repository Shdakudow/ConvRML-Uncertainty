"""Build a reviewable, fixed sensor-corruption dataset. No model is loaded.

Targets and originals are referenced, never edited or duplicated. Atomic image
writes and deterministic seeds make interrupted builds safe to resume.
"""

import argparse, hashlib, json, os, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from base_data import RAW, audit
from corruptions import corrupt, seed_for, ALL_CONDITIONS, SEVERITIES

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "dataset"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def load(path):
    with Image.open(path) as im:
        return (
            torch.from_numpy(np.array(im.convert("RGB"), copy=True))
            .permute(2, 0, 1)
            .float()
            / 255
        )


def save(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    tmp = path.with_suffix(".tmp")
    Image.fromarray(array).save(tmp, format="PNG", compress_level=3)
    tmp.replace(path)


def build_one(item):
    torch.set_num_threads(1)
    role, index, row = item
    y = load(row["measurement"]["path"])
    target_path = Path(row["target"]["path"])
    seed = seed_for(role, row["id"], 0)
    z, condition = corrupt(y, "mixed", "moderate", seed, return_condition=True)
    out = DEST / "measurements" / role / f"{index:05d}.png"
    save(out, z)
    return dict(
        id=row["id"],
        role=role,
        index=index,
        measurement=dict(path=str(out), sha256=sha(out)),
        original_measurement=row["measurement"],
        target=dict(path=str(target_path), sha256=sha(target_path)),
        corruption=condition,
        severity="moderate",
        seed=seed,
        shape=list(z.shape),
    )


def preview(roles):
    dest = ROOT / "review"
    assets = dest / "assets"
    samples = []
    selected = [
        (role, index, row)
        for role in ("train", "val", "calibration", "test")
        for index, row in enumerate(roles[role][:2])
    ]
    for n, (role, index, row) in enumerate(selected):
        y = load(row["measurement"]["path"])
        target = load(row["target"]["path"])
        seed = seed_for(role, row["id"], 0)
        sample = dict(
            id=row["id"],
            role=role,
            index=index,
            seed=seed,
            images={},
            target_sha256=sha(row["target"]["path"]),
        )
        for label, t in [("original", y), ("target", target)]:
            path = assets / f"{n:02d}_{label}.png"
            save(path, t)
            sample[label] = str(path.relative_to(dest))
        for condition in ALL_CONDITIONS:
            for severity in SEVERITIES if condition != "clean" else ("moderate",):
                path = assets / f"{n:02d}_{condition}_{severity}.png"
                z = corrupt(y, condition, severity, seed)
                save(path, z)
                sample["images"][f"{condition}:{severity}"] = str(
                    path.relative_to(dest)
                )
        mixed, actual = corrupt(y, "mixed", "moderate", seed, return_condition=True)
        path = assets / f"{n:02d}_dataset.png"
        save(path, mixed)
        sample["dataset_input"] = str(path.relative_to(dest))
        sample["dataset_corruption"] = actual
        samples.append(sample)
    write(dest / "examples.json", samples)
    payload = json.dumps(samples).replace("</", "<\\/")
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Raw DiffuserCam corruption dataset review</title><style>
body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:24px}select,button{padding:8px;margin:5px;background:#253147;color:white;border:1px solid #64748b;border-radius:5px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}figure{margin:0;background:#1f2937;padding:12px}img{width:100%;image-rendering:auto}figcaption{margin-bottom:8px}p{max-width:1000px;line-height:1.5}.note{color:#fbbf24}#status{font-family:monospace}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>Raw DiffuserCam: dataset review</h1><p>Only the sensor measurement changes. The target is the original scene image. These are data previews; no model inference or training has run for this experiment.</p>
<p class="note">Geometric warps, independent patch rotations and pixel scrambling are stress tests; they do not represent the same physical camera. Scrambling and missing pixels can irreversibly remove information.</p>
<label>Example <select id="sample"></select></label><button id="next">Next example</button><label>Corruption <select id="condition"><option value="dataset">Actual dataset input (fixed moderate mixture)</option></select></label><label>Strength <select id="severity"><option>mild</option><option selected>moderate</option><option>severe</option></select></label><p id="status"></p>
<div class="grid"><figure><figcaption>Original measurement</figcaption><img id="original"></figure><figure><figcaption>Corrupted measurement</figcaption><img id="corrupted"></figure><figure><figcaption>Original ground truth — unchanged</figcaption><img id="target"></figure><figure><figcaption>Absolute input change × 4 (display only)</figcaption><canvas id="difference" style="width:100%"></canvas></figure></div>
<script>const samples=PAYLOAD;const conditions=CONDITIONS;const S=document.querySelector('#sample'),C=document.querySelector('#condition'),V=document.querySelector('#severity');samples.forEach((s,i)=>S.add(new Option(`${i+1}: ${s.role} / ${s.id}`,i)));conditions.forEach(c=>C.add(new Option(c.replaceAll('_',' '),c)));
function update(){const s=samples[Number(S.value)];const level=C.value==='clean'?'moderate':V.value;document.querySelector('#original').src=s.original;document.querySelector('#target').src=s.target;document.querySelector('#corrupted').src=C.value==='dataset'?s.dataset_input:s.images[`${C.value}:${level}`];document.querySelector('#status').textContent=`ID ${s.id} | split ${s.role} | dataset operation ${s.dataset_corruption} | target unchanged`;draw()}
function draw(){const a=document.querySelector('#original'),b=document.querySelector('#corrupted');if(!a.complete||!b.complete||!a.naturalWidth||!b.naturalWidth)return;const c=document.querySelector('#difference');c.width=a.naturalWidth;c.height=a.naturalHeight;const ctx=c.getContext('2d');ctx.drawImage(a,0,0);const x=ctx.getImageData(0,0,c.width,c.height);ctx.drawImage(b,0,0);const y=ctx.getImageData(0,0,c.width,c.height);for(let i=0;i<x.data.length;i++)x.data[i]=i%4===3?255:Math.min(255,4*Math.abs(x.data[i]-y.data[i]));ctx.putImageData(x,0,0)}
S.onchange=C.onchange=V.onchange=update;document.querySelector('#next').onclick=()=>{S.value=(Number(S.value)+1)%samples.length;update()};document.querySelector('#original').onload=document.querySelector('#corrupted').onload=draw;update();</script></html>""".replace(
        "PAYLOAD", payload
    ).replace(
        "CONDITIONS", json.dumps(ALL_CONDITIONS)
    )
    (dest / "index.html").write_text(html)
    write(
        dest / "review_info.json",
        dict(
            samples=len(samples),
            conditions=len(ALL_CONDITIONS),
            severities=list(SEVERITIES),
            training_submitted=False,
            inference_submitted=False,
            full_dataset=str(DEST),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    counts = audit("raw")
    source = json.loads(RAW.read_text())
    roles = source["roles"]
    if args.preview_only:
        preview(roles)
        print("Review preview ready", flush=True)
        return
    DEST.mkdir(exist_ok=True)
    started = time.monotonic()
    items = [
        (role, i, row)
        for role in ("train", "val", "test", "calibration")
        for i, row in enumerate(roles[role])
    ]
    built = {role: [] for role in roles}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, result in enumerate(pool.map(build_one, items, chunksize=16), 1):
            built[result["role"]].append(result)
            if n % 250 == 0:
                write(
                    DEST / "progress.json",
                    dict(
                        processed=n,
                        total=len(items),
                        elapsed_seconds=time.monotonic() - started,
                    ),
                )
    assert {r: len(v) for r, v in built.items()} == counts
    for role in built:
        assert [r["id"] for r in built[role]] == [r["id"] for r in roles[role]]
    manifest = dict(
        protocol="fixed_raw_corruption_dataset_review_v1",
        source_split=str(RAW),
        source_split_sha256=sha(RAW),
        target_policy="Unchanged original target files; SHA256 recorded for every target; never written by this builder",
        measurement_policy="Fixed moderate mixed corruption per ID; 20% identity, otherwise uniform eight training families; quantized RGB PNG [0,255]",
        roles=built,
        counts=counts,
        corruption_counts={
            role: dict(Counter(row["corruption"] for row in rows))
            for role, rows in built.items()
        },
        held_out_test_profiles=list(ALL_CONDITIONS),
        profile_manifest="Review suite and corruptions.py define three strengths. Unseen noises are excluded from the mixed dataset.",
        elapsed_seconds=time.monotonic() - started,
        training_approval="PENDING_USER_REVIEW",
        training_submitted=False,
    )
    write(DEST / "manifest.json", manifest)
    write(
        DEST / "progress.json",
        dict(
            processed=len(items),
            total=len(items),
            status="complete",
            elapsed_seconds=time.monotonic() - started,
        ),
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "roles"}), flush=True)


if __name__ == "__main__":
    main()
