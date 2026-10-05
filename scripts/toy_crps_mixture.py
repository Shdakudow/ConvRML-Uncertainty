"""Known-truth scalar pixel-residual toy: learn mixture weights with grid CRPS.

One residual observation per feature vector. Component distributions are given
to all estimators; only the conditional weights are unknown to the learner.
This verifies the fitting mechanism, not real-image reconstruction quality.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MEANS = torch.tensor([-.14, -.025, .035, .14])
SCALES = torch.tensor([.045, .025, .065, .10])
DX = .002
GRID = torch.arange(800, dtype=torch.float32)*DX - .8 + DX/2
BASIS = .5*(1+torch.erf((GRID[None]-MEANS[:, None])/(SCALES[:, None]*2**.5)))
GRAM = BASIS@BASIS.T*DX


def true_weights(features):
    a, b = features.unbind(1)
    logits = torch.stack([-3*a+2*b, -2*b+1, 3*b-1, 3*a+b], 1)
    return logits.softmax(1)


def data(n, seed):
    gen = torch.Generator().manual_seed(seed)
    features = torch.rand(n, 2, generator=gen)*2-1
    weights = true_weights(features)
    component = torch.multinomial(weights, 1, generator=gen).squeeze(1)
    residual = MEANS[component]+SCALES[component]*torch.randn(n, generator=gen)
    observed = (GRID[None] >= residual[:, None]).float()
    return {'features': features, 'residual': residual, 'weights': weights,
            'linear': observed@BASIS.T*DX, 'constant': observed.sum(1)*DX}


class Mixture(nn.Module):
    def __init__(self, conditional):
        super().__init__()
        self.conditional = conditional
        if conditional:
            self.net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 4))
        else:
            self.logits = nn.Parameter(torch.zeros(4))

    def forward(self, features):
        logits = self.net(features) if self.conditional else self.logits.expand(len(features), -1)
        return logits.softmax(1)


def objective(weights, linear, constant):
    # Exactly the same finite-grid loss as explicitly expanding every CDF.
    return ((weights@GRAM)*weights).sum(1)-2*(weights*linear).sum(1)+constant


def fit(train, validation, conditional, seed):
    torch.manual_seed(seed)
    model = Mixture(conditional)
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    generator = torch.Generator().manual_seed(seed+100)
    best = float('inf'); saved = None; chosen_epoch = None
    start = time.perf_counter()
    for epoch in range(1, 121):
        order = torch.randperm(len(train['features']), generator=generator)
        for ix in order.split(256):
            loss = objective(model(train['features'][ix]), train['linear'][ix], train['constant'][ix]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val = objective(model(validation['features']), validation['linear'], validation['constant']).mean().item()
        if val < best:
            best = val; saved = copy.deepcopy(model.state_dict()); chosen_epoch = epoch
    model.load_state_dict(saved)
    return model, {'best_validation_crps': best, 'selected_epoch': chosen_epoch,
                   'training_seconds': time.perf_counter()-start,
                   'parameters': sum(p.numel() for p in model.parameters())}


def evaluate(weights, test):
    score = objective(weights, test['linear'], test['constant'])
    diff = weights-test['weights']
    true_cdf_error = ((diff@GRAM)*diff).sum(1).clamp_min(0)
    cdf = weights@BASIS
    levels = torch.tensor([.025, .1, .25, .5, .75, .9, .975])
    indices = torch.searchsorted(cdf.contiguous(), levels.expand(len(cdf), -1).contiguous()).clamp(max=len(GRID)-1)
    quantiles = GRID[indices]
    observed = (test['residual'][:, None] <= quantiles).float().mean(0)
    return {'approx_crps': score.mean().item(), 'integrated_squared_error_to_true_cdf': true_cdf_error.mean().item(),
            'coverage95': ((test['residual'] >= quantiles[:, 0]) & (test['residual'] <= quantiles[:, -1])).float().mean().item(),
            'width95': (quantiles[:, -1]-quantiles[:, 0]).mean().item(),
            'quantile_calibration_mae': (observed-levels).abs().mean().item()}, score


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    start = time.perf_counter(); results = []; example_models = {}
    for seed in [11, 22, 33]:
        train, validation, test = [data(n, seed*100+offset) for n, offset in [(4096, 0), (1024, 1), (4096, 2)]]
        # Verify the fast quadratic objective against the direct CDF definition.
        w = train['weights'][:16]
        direct = ((w@BASIS-(GRID[None] >= train['residual'][:16, None]).float())**2).sum(1)*DX
        assert torch.allclose(direct, objective(w, train['linear'][:16], train['constant'][:16]), atol=3e-7)
        run = {'seed': seed, 'metrics': {}, 'training': {}}; scores = {}
        for name, conditional in [('global', False), ('conditional', True)]:
            model, metadata = fit(train, validation, conditional, seed)
            with torch.no_grad():
                run['metrics'][name], scores[name] = evaluate(model(test['features']), test)
            run['training'][name] = metadata
            torch.save(model.state_dict(), args.out/f'{name}_seed{seed}.pt')
            if seed == 11: example_models[name] = model
        run['metrics']['oracle'], _ = evaluate(test['weights'], test)
        delta = (scores['conditional']-scores['global']).numpy()
        indices = np.random.default_rng(seed).integers(0, len(delta), (1000, len(delta)))
        run['paired_crps_difference'] = {'conditional_minus_global': float(delta.mean()),
                                       'bootstrap95': np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
        results.append(run); print(json.dumps(run), flush=True)
    average = {name: {key: float(np.mean([r['metrics'][name][key] for r in results]))
                     for key in results[0]['metrics'][name]} for name in ['global', 'conditional', 'oracle']}
    protocol = ('Synthetic independent scalar pixel residuals with two observable features. Four known Gaussian components, feature-dependent true weights. '
                'Global weights versus a 116-parameter conditional MLP, both fitted with identical approximate CRPS and data. '
                '4096 training / 1024 validation / 4096 test observations per seed; three independently generated datasets. '
                '120 epochs, batch 256; select checkpoint by validation loss only. One observed residual per feature vector; true weights never enter training loss. '
                'Midpoint quadrature on [-0.8,0.8], spacing .002; omitted tails are negligible for this specified generator, but this is finite-grid CRPS. '
                'Intervals use the same grid. No calibration or reconstruction retraining. Known component library makes this an intentionally favorable mechanism test, not evidence of real-image improvement.')
    report = {'protocol': protocol, 'runs': results, 'average': average, 'total_seconds': time.perf_counter()-start}
    (args.out/'results.json').write_text(json.dumps(report, indent=2))
    lines = ['# CRPS mixture toy test', '', protocol, '', '| Estimator | Approximate CRPS | True-CDF squared error | Coverage95 | Width95 |', '|---|---:|---:|---:|---:|']
    for name, r in average.items():
        lines.append(f"| {name} | {r['approx_crps']:.6f} | {r['integrated_squared_error_to_true_cdf']:.6f} | {r['coverage95']:.2%} | {r['width95']:.4f} |")
    lines += ['', f"Total CPU experiment time: {report['total_seconds']:.1f} seconds."]
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    features = torch.tensor([[-.8, -.8], [.8, -.8], [-.8, .8], [.8, .8]])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    with torch.no_grad():
        curves = {'True CDF': true_weights(features)@BASIS,
                  'Global mixture': example_models['global'](features)@BASIS,
                  'CRPS conditional mixture': example_models['conditional'](features)@BASIS}
    for i, ax in enumerate(axes.flat):
        for label, cdf in curves.items(): ax.plot(GRID.numpy(), cdf[i].numpy(), label=label)
        ax.set_title(f'Observable features: ({features[i,0]:.1f}, {features[i,1]:.1f})'); ax.set_xlim(-.4, .5); ax.set_ylim(0, 1)
        ax.set_xlabel('Residual'); ax.set_ylabel('Cumulative probability'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Known-truth toy: learning pixel residual distributions with CRPS')
    fig.tight_layout(); fig.savefig(args.out/'cdf_comparison.png', dpi=170); fig.savefig(args.out/'cdf_comparison.pdf'); plt.close(fig)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
