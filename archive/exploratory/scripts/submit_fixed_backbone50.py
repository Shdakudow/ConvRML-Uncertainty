"""Submit serial two-epoch stages; failures block subsequent stages."""
import json, subprocess
from pathlib import Path
root = Path('results/fixed_backbone50_20260927')
root.mkdir(parents=True, exist_ok=True)
record = root/'jobs.json'
assert not record.exists(), 'Already submitted; inspect jobs before resubmitting'
sb = '/usr/local/slurm/current/bin/sbatch'
jobs = {'training': []}
def submit(args):
    return subprocess.check_output([sb, '--parsable', *args], text=True).strip().split(';')[0]
def save():
    record.write_text(json.dumps(jobs, indent=2))
previous = submit(['--job-name=fixed-b50-prep', '--partition=monakhova', '--constraint=avx2', '--cpus-per-task=2', '--mem=12G', '--time=00:20:00', '--output=logs/fixed_b50_prep_%j.out', '--wrap', 'cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/fixed_backbone50.py prepare'])
jobs['prepare'] = previous
save()
for kind, epochs in [('normal', range(32, 51, 2)), ('fixed', range(2, 51, 2))]:
    for epoch in epochs:
        previous = submit(['--dependency=afterok:'+previous, '--job-name=fixed-b50-'+kind+'-'+str(epoch), 'scripts/fixed_backbone50.sbatch', 'train', '--kind', kind, '--epochs', str(epoch)])
        jobs['training'].append({'kind':kind, 'end_epoch':epoch, 'job':previous})
        save()
previous = submit(['--dependency=afterok:'+previous, 'scripts/fixed_backbone50.sbatch', 'infer'])
jobs['inference'] = previous
save()
jobs['report'] = submit(['--dependency=afterok:'+previous, '--job-name=fixed-b50-report', '--partition=monakhova', '--constraint=avx2', '--cpus-per-task=4', '--mem=16G', '--time=01:00:00', '--output=logs/fixed_b50_report_%j.out', '--wrap', 'cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/fixed_backbone50.py report'])
save()
print(json.dumps(jobs, indent=2))
