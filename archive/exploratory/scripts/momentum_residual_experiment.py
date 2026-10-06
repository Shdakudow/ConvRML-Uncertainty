"""Compare cumulative and exponentially weighted context residual histograms.

Frozen predictions only. This estimates marginal residual distributions, not
Adam gradient uncertainty. Test labels are used exclusively for evaluation.
"""
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.context_residual_experiment import collect, context, sampled_metrics, finalize
from scripts.residual_distribution_experiment import arrays, widths, transform, LEVELS, TD, CD, SOURCE, sha

ROOT = Path('results/momentum_residual_20260925')


def invert_hist(probability, edges):
    probability = probability / probability.sum()
    # Invert within occupied bins with a piecewise-uniform density.
    cdf = np.cumsum(probability)
    bins = np.minimum(np.searchsorted(cdf, LEVELS), len(probability)-1)
    previous = np.r_[0., cdf[:-1]][bins]
    fraction = (LEVELS-previous) / np.maximum(probability[bins], 1e-30)
    return edges[bins] + fraction * np.diff(edges)[bins]


def fit_all(fields):
    cuts = np.quantile(fields['context'], [1/3, 2/3])
    group = np.searchsorted(cuts, fields['context'])
    error = fields['x']-fields['r']
    z = error/np.where(error < 0, fields['wl'], fields['wu'])
    # Asinh spacing preserves central resolution while retaining every fit residual.
    radius = max(float(np.max(np.abs(z))), 1.) * 1.000001
    edges = np.sinh(np.linspace(-np.arcsinh(radius), np.arcsinh(radius), 257))
    indices = np.clip(np.searchsorted(edges, z, side='right')-1, 0, 255)
    # Per-image counts permit reproducible image shuffles without resampling pixels.
    hist = np.zeros((len(z), 4, 256), dtype=np.float64)
    for i in range(len(z)):
        hist[i, 0] = np.bincount(indices[i], minlength=256)
        for g in range(3):
            hist[i, g+1] = np.bincount(indices[i][group[i] == g], minlength=256)

    def table(counts):
        global_p = counts[0]/counts[0].sum()
        tables = []
        for g in range(3):
            p = counts[g+1]/counts[g+1].sum() if counts[g+1].sum() else global_p
            tables.append(invert_hist(.75*p+.25*global_p, edges).tolist())
        return {'cuts': cuts.tolist(), 'global': invert_hist(global_p, edges).tolist(), 'context': tables}

    fits = {'cumulative': table(hist.sum(0))}
    for beta in [.9, .99]:
        for seed in [11, 22, 33]:
            order = np.random.default_rng(seed).permutation(len(z))
            state = np.zeros((4, 256))
            for start in range(0, len(z), 8):
                state = beta*state + (1-beta)*hist[order[start:start+8]].sum(0)
            # Normalization cancels the common EMA initialization-bias factor.
            fits[f'ema_{beta}_seed{seed}'] = table(state)
    return fits, {'bin_edges': edges.tolist(), 'updates': len(z)//8}


def test_fields(method):
    pred = arrays(method, 'test')
    pairs = np.load(TD/'pairs.npy', mmap_mode='r')
    rng = np.random.default_rng(4203)
    for s in range(0, 1000, 8):
        e = min(s+8, 1000); n = e-s
        r, lo, hi = [pred[k][s:e] for k in ['r', 'lo', 'hi']]
        wl, wu = widths(r, lo, hi)
        x = pairs[22500+s:22500+e, 1].astype(np.float32)/255
        indices = rng.integers(0, 3*256*256, (n, 4096))
        fields = {key: np.take_along_axis(v.reshape(n, -1), indices, axis=1)
                  for key, v in zip(['r', 'wl', 'wu', 'x'], [r, wl, wu, x])}
        fields['context'] = np.take_along_axis(context(r).reshape(n, -1), indices % (256*256), axis=1)
        yield fields


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    source = json.loads((SOURCE/'inference.json').read_text())
    assert source['test_manifest_sha256'] == sha(TD/'manifest.json')
    assert source['calibration_manifest_sha256'] == sha(CD/'manifest.json')
    report = {'protocol': 'Frozen uniform/endpoints pilot models, not the four epoch-30 models. Fit first 1024 separate calibration images, 4096 sampled RGB values/image. Three reconstruction-context groups, 256 bins, 75% group + 25% global CDF. EMA batch size 8, beta 0.9/0.99, three image-order seeds. No post-calibration or test-based selection. Evaluate identical 4096 sampled RGB values on each of 1000 test images. Coverage/width are sampled estimates of nominal 95% intervals. Integrated score covers q=.005 to .995, not full CRPS. Bootstrap assumes independent images. Remaining 1024 calibration images are unused.', 'source': source, 'results': {}}
    for method in ['uniform', 'endpoints']:
        start = time.perf_counter()
        fields = collect(method, 0, 1024, 4201)
        fit_start = time.perf_counter()
        fits, metadata = fit_all(fields)
        fit_seconds = time.perf_counter()-fit_start
        del fields
        (ROOT/f'{method}_frozen_fit.json').write_text(json.dumps({'fits': fits, 'metadata': metadata, 'fit_seconds_all_variants': fit_seconds}, indent=2))
        accum = {name: {'hit': np.zeros((3, len(LEVELS))), 'loss': np.zeros((3, len(LEVELS))), 'counts': np.zeros(3), 'pb': [], 'coverage': [], 'width': [], 'seconds': 0.} for name in fits}
        for batch, fields in enumerate(test_fields(method)):
            for name, fit in fits.items():
                ac = accum[name]; tick = time.perf_counter()
                hit, loss, counts, pb = sampled_metrics(fields, fit, 'context')
                ac['hit'] += hit; ac['loss'] += loss; ac['counts'] += counts; ac['pb'].append(pb)
                group = np.searchsorted(fit['cuts'], fields['context'])
                q = np.array(fit['context'])
                lo = transform(fields['r'], fields['wl'], fields['wu'], q[group, 2])
                hi = transform(fields['r'], fields['wl'], fields['wu'], q[group, -3])
                assert np.all(lo <= hi)
                ac['coverage'].extend(((lo <= fields['x']) & (fields['x'] <= hi)).mean(1).tolist())
                ac['width'].extend((hi-lo).mean(1).tolist())
                ac['seconds'] += time.perf_counter()-tick
            if batch % 25 == 0: print(method, 'evaluated', (batch+1)*8, flush=True)
        baseline = 2*np.trapz(np.concatenate(accum['cumulative']['pb']), LEVELS, axis=1)
        results = {}
        for name, ac in accum.items():
            pb = np.concatenate(ac['pb'])
            result = finalize(ac['hit'], ac['loss'], ac['counts'], pb)
            scores = 2*np.trapz(pb, LEVELS, axis=1)
            delta = scores-baseline
            boot = np.random.default_rng(4204).integers(0, 1000, (1000, 1000))
            result.update(coverage95=float(np.mean(ac['coverage'])), width95=float(np.mean(ac['width'])),
                          score_delta_vs_cumulative=float(delta.mean()), paired_image_bootstrap95=np.quantile(delta[boot].mean(1), [.025, .975]).tolist(),
                          scoring_seconds=ac['seconds'])
            results[name] = result
            np.save(ROOT/f'{method}_{name}_per_image_scores.npy', scores)
        report['results'][method] = {'variants': results, 'fit_seconds_all_variants': fit_seconds, 'total_seconds': time.perf_counter()-start}
        (ROOT/'comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# Momentum residual histogram experiment', '', report['protocol'], '', '| Model | Estimator | Coverage95 | Width95 | Integrated score | Quantile calibration MAE |', '|---|---|---:|---:|---:|---:|']
    for method, result in report['results'].items():
        for name, v in result['variants'].items():
            lines.append(f"| {method} | {name} | {v['coverage95']:.2%} | {v['width95']:.6f} | {v['truncated_integrated_quantile_score']:.6f} | {v['quantile_calibration_mae']:.5f} |")
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
