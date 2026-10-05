"""Isolated epoch-25 RML continuation: final-only versus everywhere FiLM."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from models.shamus_rml import ShamusRML
from models.formal_quantile import QuantileRML, pinball
from models.film_everywhere import EverywhereQuantileRML
from scripts.formal_rml_train import Pairs
from scripts.training_records import manifest_digest, save_milestone

ROOT = Path('results/film_everywhere_rml_e25_20261001')
DATA = Path('data/optics_intervals_60k_5k_5k_20260928')
CAL = Path('data/formal_rml_calibration95')
SEED = 20261001


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, content):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(content, indent=2))
    tmp.replace(path)


def make_model(variant, state):
    if variant == 'everywhere':
        model = EverywhereQuantileRML(ShamusRML())
        model.load_original(state)
    else:
        model = QuantileRML(ShamusRML(), 'conditioned', levels=(.05, .95))
        model.load_state_dict(state, strict=True)
    return model.cuda()


def quantile(model, y, q):
    if isinstance(model, EverywhereQuantileRML):
        return model.predict(y, q=q)
    f = model.base.up2(model.base.features(y))
    return model.predict(y, f, q)


def train_step(model, opt, batch, qrng):
    y, x = batch[:, 0].cuda(), batch[:, 1].cuda()
    opt.zero_grad(set_to_none=True)
    # Accumulate the same MSE + pinball gradients with sequential graphs to
    # avoid holding two full-backbone activation graphs at once. Both arms use
    # this schedule. No parameter update happens between the two backwards.
    with torch.autocast('cuda', dtype=torch.bfloat16):
        mse = (model.base(y).float()-x).square().mean()
    if not torch.isfinite(mse):
        raise RuntimeError('Nonfinite reconstruction loss')
    mse.backward()
    q = torch.rand(len(y), device='cuda', generator=qrng)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        qloss = pinball(quantile(model, y, q).float(), x, q)
    if not torch.isfinite(qloss):
        raise RuntimeError('Nonfinite pinball loss')
    qloss.backward()
    if isinstance(model, EverywhereQuantileRML):
        if any(f.affine.weight.grad is None for f in model.deep_films.values()):
            raise RuntimeError('A deep FiLM layer is disconnected')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return float(mse), float(qloss), float(norm)


@torch.inference_mode()
def validate(model, loader):
    model.eval()
    sums = dict(mse=0., pinball=0., psnr=0., ssim=0.)
    count = 0
    for b in loader:
        y, x = b[:, 0].cuda(), b[:, 1].cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            r, lo, hi = [v.float() for v in model.bounds(y)]
        loss = (pinball(lo, x, x.new_full((len(y),), .05)) +
                pinball(hi, x, x.new_full((len(y),), .95))) / 2
        mse = (r-x).square().mean((1, 2, 3))
        sums['mse'] += float(mse.sum())
        sums['pinball'] += float(loss)*len(y)
        sums['psnr'] += float((-10*mse.clamp_min(1e-12).log10()).sum())
        sums['ssim'] += float(ssim(r, x, data_range=1., reduction='none').sum())
        count += len(y)
    result = {k: v/count for k, v in sums.items()}
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError('Nonfinite validation metric')
    return result


@torch.inference_mode()
def evaluate(model, out, epoch, metadata):
    model.eval()
    target = out/'evaluation'/f'epoch_{epoch:03d}'
    target.mkdir(parents=True, exist_ok=True)
    cal = np.load(CAL/'pairs.npy', mmap_mode='r')
    score_path = target/'scores.tmp.npy'
    scores = np.lib.format.open_memmap(score_path, mode='w+', dtype=np.float32,
                                       shape=(len(cal), *cal.shape[2:]))
    def predictions(b):
        y, x = b[:, 0].cuda(), b[:, 1].cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            r, lo, hi = [v.float() for v in model.bounds(y)]
        if not all(torch.isfinite(v).all() for v in (r, lo, hi)):
            raise RuntimeError('Nonfinite prediction')
        return x, r, lo, hi, (r-lo).clamp_min(1e-6), (hi-r).clamp_min(1e-6)
    for start in range(0, len(cal), 2):
        b = torch.from_numpy(np.array(cal[start:start+2], copy=True)).float()/255
        x, r, lo, hi, wl, wu = predictions(b)
        scores[start:start+len(x)] = torch.maximum((r-x)/wl, (x-r)/wu).cpu().numpy()
    scores.flush()
    factor = max(0., float(np.quantile(scores, .9, method='higher', overwrite_input=True)))
    del scores
    score_path.unlink()
    rows = []
    for b in DataLoader(Pairs('test', DATA), batch_size=2, num_workers=2):
        x, r, lo, hi, wl, wu = predictions(b)
        lower, upper = r-factor*wl, r+factor*wu
        for i in range(len(x)):
            e = (r[i]-x[i]).abs()
            rows.append(dict(test_index=len(rows), psnr=-10*math.log10(max(float(e.square().mean()), 1e-12)),
                             ssim=float(ssim(r[i:i+1], x[i:i+1], data_range=1.)),
                             mae=float(e.mean()), coverage=float(((x[i]>=lower[i])&(x[i]<=upper[i])).float().mean()),
                             width=float((upper[i]-lower[i]).mean()),
                             native_coverage=float(((x[i]>=lo[i])&(x[i]<=hi[i])).float().mean()),
                             native_crossing_fraction=float((lo[i]>hi[i]).float().mean())))
    with (target/'per_image.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = dict(**metadata, epoch=epoch, alpha=.1, factor=factor,
                  calibration_count=len(cal), test_count=len(rows),
                  calibration_sha256=manifest_digest(CAL),
                  calibration='Empirical pooled-pixel scaling; no image-level guarantee',
                  metrics={k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='test_index'})
    write_json(target/'metrics.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['smoke', 'train', 'evaluate'])
    parser.add_argument('--variant', choices=['control', 'everywhere'], required=True)
    parser.add_argument('--epochs', type=int, default=2, help='Additional epochs, not total backbone epochs')
    a = parser.parse_args()
    if a.epochs not in (2, 4):
        raise ValueError('This controlled pilot is limited to +2 and +4 epochs')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    source = ROOT/'source/epoch_025.pt'
    source_hash = sha(source)
    provenance = json.loads((ROOT/'source/provenance.json').read_text())
    if source_hash != provenance['sha256']:
        raise ValueError('Copied source checkpoint changed')
    ck = torch.load(source, map_location='cpu', weights_only=False)
    manifest = manifest_digest(DATA)
    if ck['epoch'] != 25 or ck['manifest_sha256'] != manifest or tuple(ck['levels']) != (.05, .95):
        raise ValueError('Unexpected source checkpoint/data/quantiles')
    model = make_model(a.variant, ck['model'])
    del ck
    metadata = dict(variant=a.variant, source_epoch=25, source_sha256=source_hash,
                    manifest_sha256=manifest, learning_rate=1e-5, weight_decay=1e-4,
                    batch_size=2, seed=SEED, optimizer='fresh AdamW in both arms',
                    train_count=len(Pairs('train', DATA)), val_count=len(Pairs('val', DATA)),
                    film_locations=getattr(model, 'film_locations', []))
    out = ROOT/a.variant
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # Reset RNG after architecture construction so batches and q draws match.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    qrng = torch.Generator(device='cuda').manual_seed(SEED+1)
    drng = torch.Generator().manual_seed(SEED+2)
    start = 0
    history = []
    if a.action == 'evaluate':
        saved = torch.load(out/'checkpoints'/f'epoch_{25+a.epochs:03d}.pt', map_location='cpu', weights_only=False)
        if saved['source_sha256'] != source_hash or saved['manifest_sha256'] != manifest:
            raise ValueError('Evaluation checkpoint provenance mismatch')
        model.load_state_dict(saved['model'], strict=True)
        del saved
        evaluate(model, out, 25+a.epochs, metadata)
        return
    if a.action == 'train' and (out/'latest.pt').exists():
        saved = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        if saved['source_sha256'] != source_hash or saved['manifest_sha256'] != manifest:
            raise ValueError('Resume provenance mismatch')
        model.load_state_dict(saved['model'], strict=True)
        opt.load_state_dict(saved['optimizer'])
        start, history = saved['additional_epochs'], saved['history']
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        qrng.set_state(saved['q_rng']); drng.set_state(saved['data_rng'])
        del saved
        # Recover an interrupted save after atomic latest.pt publication.
        if not (out/'checkpoints'/f'epoch_{25+start:03d}.pt').exists():
            save_milestone(out, model, 25+start, 1, **metadata, history=history, levels=(.05, .95))
    train = DataLoader(Pairs('train', DATA), batch_size=2, shuffle=True, generator=drng,
                       num_workers=2, pin_memory=True)
    val = DataLoader(Pairs('val', DATA), batch_size=2, num_workers=2, pin_memory=True)
    if a.action == 'smoke':
        model.eval()
        b = next(iter(val))
        y = b[:, 0].cuda()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            f, r = model.features(y)
            max_error = 0.
            for qval in (.05, .5, .95):
                q = torch.full((len(y),), qval, device='cuda')
                original = QuantileRML.predict(model, y, f, q)
                modified = model.predict(y, f, q)
                torch.testing.assert_close(original, modified, rtol=0, atol=0)
                max_error = max(max_error, float((original-modified).abs().max()))
        del f, r, original, modified
        model.train()
        for i, b in enumerate(train):
            train_step(model, opt, b, qrng)
            if i == 4:
                torch.cuda.synchronize(); begin = time.monotonic()
            if i == 14:
                break
        torch.cuda.synchronize()
        batch_seconds = (time.monotonic()-begin)/10
        model.eval()
        begin = time.monotonic()
        with torch.inference_mode():
            for i, b in enumerate(val):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    values = model.bounds(b[:, 0].cuda())
                    assert all(torch.isfinite(v).all() for v in values)
                if i == 9:
                    break
        torch.cuda.synchronize()
        val_seconds = (time.monotonic()-begin)/10
        estimate = 2*(batch_seconds*len(train)+val_seconds*len(val))*1.3
        result = dict(**metadata, passed=estimate < 4.8*3600, identity_max_error=max_error,
                      seconds_per_train_batch=batch_seconds, seconds_per_validation_batch=val_seconds,
                      estimated_two_epoch_seconds_with_margin=estimate,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name())
        write_json(out/'smoke.json', result)
        print(json.dumps(result), flush=True)
        if not result['passed']:
            raise RuntimeError('Smoke runtime exceeds five-hour allocation')
        return
    smoke = json.loads((out/'smoke.json').read_text())
    if not smoke['passed'] or smoke['source_sha256'] != source_hash:
        raise RuntimeError('Missing/passed smoke gate or source mismatch')
    if not 0 <= a.epochs-start <= 2:
        raise ValueError('Each job may advance at most two epochs')
    write_json(out/'config.json', metadata)
    for epoch in range(start, a.epochs):
        model.train(); begin = time.monotonic(); mse_sum = q_sum = 0.
        for b in train:
            mse, qloss, _ = train_step(model, opt, b, qrng)
            mse_sum += mse*len(b); q_sum += qloss*len(b)
        validation = validate(model, val)
        row = dict(epoch=26+epoch, additional_epochs=epoch+1, train_mse=mse_sum/len(train.dataset),
                   train_pinball=q_sum/len(train.dataset), validation=validation, seconds=time.monotonic()-begin)
        history.append(row)
        # Publish the resume state first; missing immutable weights can then be
        # recovered from exactly that state after interruption.
        torch.save(dict(**metadata, model=model.state_dict(), optimizer=opt.state_dict(),
                        additional_epochs=epoch+1, history=history, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(), q_rng=qrng.get_state(), data_rng=drng.get_state()),
                   out/'latest.tmp')
        (out/'latest.tmp').replace(out/'latest.pt')
        save_milestone(out, model, 26+epoch, 1, **metadata, history=history, levels=(.05, .95))
        write_json(out/'history.json', history)
        print(json.dumps(row), flush=True)

if __name__ == '__main__':
    main()
