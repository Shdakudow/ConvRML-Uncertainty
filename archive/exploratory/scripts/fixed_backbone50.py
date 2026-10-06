"""Corrected fixed-head experiment: exact normal epoch 50, then fresh head 50."""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts import four_models50 as evaluation
ROOT = Path('results/fixed_backbone50_20260927')
SOURCE = Path('results/normal_convrml_epoch30_20260924')
OLD = evaluation.ROOT

def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'prepared.json').exists():
        return
    checkpoint = torch.load(SOURCE/'normal/latest.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 30
    assert checkpoint['manifest_sha256'] == evaluation.fingerprint(evaluation.TD/'manifest.json')
    del checkpoint
    (ROOT/'normal').mkdir(exist_ok=True)
    for name in ('latest.pt', 'best.pt', 'history.json'):
        shutil.copy2(SOURCE/'normal'/name, ROOT/'normal'/name)
    shutil.copy2(SOURCE/'smoke_normal.json', ROOT/'smoke_normal.json')
    protocol = json.loads((OLD/'prepared.json').read_text())
    protocol.update(backbone_source=str(SOURCE/'normal/latest.pt'), backbone_epochs=50,
                    fresh_head_epochs=50, head_initialization='Copy epoch-50 reconstruction head into two new quantile outputs; fresh optimizer',
                    fixed_semantics='50 MSE reconstruction epochs; freeze exact epoch 50; 50 fresh endpoint-head epochs',
                    inference_checkpoint='exact backbone 50 and exact head 50',
                    other_methods='Reuse original completed epoch-50 predictions without further training')
    (ROOT/'prepared.json').write_text(json.dumps(protocol, indent=2))

def train(kind, epoch):
    command = [sys.executable, '-u', 'scripts/formal_quantile_train.py', kind,
               '--epochs', str(epoch), '--root', str(ROOT)]
    if kind == 'fixed':
        ck = torch.load(ROOT/'normal/epoch_50.pt', map_location='cpu', weights_only=False)
        assert ck['epoch'] == 50 and ck['trained_stage'] == 'normal'
        del ck
        command += ['--backbone-checkpoint', str(ROOT/'normal/epoch_50.pt')]
    subprocess.run(command, check=True)
    if epoch in (40, 50):
        ck = torch.load(ROOT/kind/'latest.pt', map_location='cpu', weights_only=False)
        assert ck['epoch'] == epoch
        snapshot = {k: ck[k] for k in ('model', 'epoch', 'manifest_sha256')}
        snapshot['trained_stage'] = kind
        torch.save(snapshot, ROOT/kind/f'epoch_{epoch}.pt')

def configure_evaluation():
    evaluation.ROOT = ROOT
    evaluation.QR = ROOT
    evaluation.PR = ROOT/'predictions'
    evaluation.PR.mkdir(exist_ok=True)
    evaluation.METHODS['fixed'] = 'Fixed head (backbone 50 + fresh head 50)'
    for worker in (0, 1, 2, 3, 4, 5, 7):
        dest = evaluation.PR/f'worker_{worker}'
        if not dest.exists():
            dest.symlink_to((OLD/'predictions'/f'worker_{worker}').resolve(), target_is_directory=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'train', 'infer', 'report'])
    p.add_argument('--kind', choices=['normal', 'fixed'])
    p.add_argument('--epochs', type=int)
    a = p.parse_args()
    if a.action == 'prepare':
        prepare()
    elif a.action == 'train':
        train(a.kind, a.epochs)
    else:
        configure_evaluation()
        if a.action == 'infer':
            evaluation.infer(6)
        else:
            evaluation.report()
            report = ROOT/'comparison.md'
            text = report.read_text().replace(
                'Fixed-head epoch 50 means 50 head-training epochs on its original frozen normal backbone (which had 20 reconstruction-training epochs).',
                'CORRECTED fixed head: reconstruction backbone trained for 50 epochs, frozen at exact epoch 50, then a fresh endpoint head trained for 50 epochs. Other methods reuse their earlier epoch-50 results. This gives fixed head additional total training; it is not a matched-compute comparison.')
            report.write_text(text)
            subprocess.run([sys.executable, 'scripts/render_uncertainty_comparison.py', '--input',
                            str(ROOT/'examples/examples.npz'), '--out', str(ROOT/'visuals')], check=True)
