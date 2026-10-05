"""Expand training to 60k without moving existing held-out or calibration pairs."""
import json
import random
import re
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.prepare_formal_rml import image, ROOT

OUT = Path('data/optics_intervals_60k_5k_5k_20260928')
COUNTS = {'train': 60000, 'val': 5000, 'test': 5000}
OFFSETS = {'train': 0, 'val': 60000, 'test': 65000}


def extension_slots(original):
    # Reserve new held-out examples before adding the new training examples.
    for role in ('val', 'test', 'train'):
        for index in range(len(original[role]), COUNTS[role]):
            yield role, OFFSETS[role] + index


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        existing=json.loads((OUT/'manifest.json').read_text())
        if any(len(existing[role])!=count for role,count in COUNTS.items()):
            raise ValueError('Existing manifest has an incompatible split')
        print('Already prepared', flush=True)
        return
    original = json.loads(Path('data/formal_rml_25k/manifest.json').read_text())
    calibration = json.loads(Path('data/formal_rml_calibration95/manifest.json').read_text())['calibration']
    reserved = original['train'] + original['val'] + original['test'] + calibration
    ids_seen = {r['id'] for r in reserved}
    hashes_seen = {r['target_sha256'] for r in reserved}
    if len(hashes_seen) != len(reserved):
        raise ValueError('Existing splits contain duplicate processed targets')
    old = np.load('data/formal_rml_25k/pairs.npy', mmap_mode='r')
    cache = np.lib.format.open_memmap(OUT / 'pairs.npy', mode='w+', dtype=np.uint8,
                                     shape=(70000, 2, 3, 256, 256))
    cache[:20000] = old[:20000]
    cache[60000:62500] = old[20000:22500]
    cache[65000:67500] = old[22500:25000]
    records = {role:list(original[role]) for role in COUNTS}
    slots = iter(extension_slots(original))
    prepared = 0
    needed = sum(COUNTS.values()) - sum(len(v) for v in records.values())
    candidates = []
    for path in (ROOT / 'rml').glob('4x_img_*_cam_1.tiff'):
        match = re.fullmatch(r'4x_img_(\d+)_cam_1.tiff', path.name)
        if match:
            ident = int(match[1])
            if ident not in ids_seen and ident not in (5, 6):
                candidates.append(ident)
    candidates.sort()
    random.Random(20260928).shuffle(candidates)
    rejected = []
    for ident in candidates:
        yp = ROOT / 'rml' / f'4x_img_{ident}_cam_1.tiff'
        xp = ROOT / 'gt2rml' / f'warped_4x_undistorted_img_{ident}_cam_2.tiff'
        try:
            x = image(xp)
            digest = hashlib.sha256(x.tobytes()).hexdigest()
            if digest in hashes_seen:
                rejected.append({'id': ident, 'reason': 'duplicate processed target'})
                continue
            y = image(yp)
        except Exception as exc:
            rejected.append({'id': ident, 'reason': str(exc)})
            continue
        role,position = next(slots)
        cache[position, 0] = y
        cache[position, 1] = x
        hashes_seen.add(digest)
        records[role].append({'id': ident, 'measurement': str(yp), 'target': str(xp),
                        'target_sha256': digest})
        prepared += 1
        if prepared % 1000 == 0:
            cache.flush()
            print('PREPARED', {role:len(rows) for role,rows in records.items()}, flush=True)
        if prepared == needed:
            break
    if any(len(records[role]) != count for role,count in COUNTS.items()):
        raise RuntimeError(f'Insufficient unique pairs: {prepared}/{needed} additional pairs')
    cache.flush()
    report = {**records,
              'seed': 20260928, 'rejected': rejected,
              'calibration_manifest': 'data/formal_rml_calibration95/manifest.json',
              'notes': ['Existing validation/test membership preserved and expanded to 5000 each; calibration unchanged.',
                        'Exact processed target hashes disjoint; acquisition groups and near duplicates not certified.',
                        'Previously inspected test images are not a new untouched test set.']}
    temporary = OUT / 'manifest.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(OUT / 'manifest.json')
    print('READY 60000 train / 5000 val / 5000 test / separate 2048 calibration', flush=True)


if __name__ == '__main__':
    main()
