"""Isolated 60k study using the audited resumable trainers."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path('results/optics_intervals_60k_5k_5k_20260928')
DATA = Path('data/optics_intervals_60k_5k_5k_20260928')
CALIBRATION = Path('data/formal_rml_calibration95')


def command(worker, epoch, smoke=False):
    if worker < 6:
        args = [sys.executable, '-u', 'scripts/formal_rml_train.py',
                '--worker', str(worker), '--epochs', str(epoch), '--out', str(ROOT/'baselines'),
                '--patience', '0', '--max-new-epochs', '2', '--skip-inference']
    else:
        args = [sys.executable, '-u', 'scripts/formal_quantile_train.py',
                'normal' if worker == 6 else 'conditioned', '--epochs', str(epoch),
                '--root', str(ROOT/'quantiles'), '--interval-alpha', '.10', '--isolated-q']
    args += ['--data', str(DATA), '--save-every', '5']
    if smoke:
        args += ['--smoke']
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['smoke', 'train', 'head', 'evaluate', 'plot'])
    p.add_argument('--worker', type=int, default=0, choices=range(8))
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--backbone-epoch', type=int)
    a = p.parse_args()
    if a.stage == 'head':
        if a.backbone_epoch not in range(5,51,5):
            raise ValueError('Expected a 5-epoch backbone milestone')
        subprocess.run([sys.executable, '-u', 'scripts/formal_quantile_train.py', 'fixed',
                        '--epochs', str(a.epochs), '--root', str(ROOT/'heads'/f'backbone_{a.backbone_epoch:03d}'),
                        '--backbone-checkpoint', str(ROOT/'quantiles/normal/checkpoints'/f'epoch_{a.backbone_epoch:03d}.pt'),
                        '--interval-alpha', '.10', '--data', str(DATA), '--save-every', '5'], check=True)
    elif a.stage in ('evaluate','plot'):
        import large_interval_evaluate as evaluation
        evaluation.ROOT=ROOT
        evaluation.DATA=DATA
        evaluation.CALIBRATION=CALIBRATION
        if a.stage=='plot':evaluation.plot()
        else:evaluation.evaluate(a.worker,a.backbone_epoch)
    else:
        subprocess.run(command(a.worker, a.epochs, a.stage == 'smoke'), check=True)


if __name__ == '__main__':
    main()
