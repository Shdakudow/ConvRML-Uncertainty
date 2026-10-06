"""Copy immutable source weights and queue two matched fine-tuning arms."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
ROOT = Path('results/film_everywhere_rml_e25_20261001')
SOURCE = Path('results/optics_intervals_60k_5k_5k_20260928/quantiles/conditioned/checkpoints/epoch_025.pt')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    record = ROOT/'jobs.json'
    if record.exists():
        raise FileExistsError('Already submitted or partially submitted; inspect jobs.json')
    source_dir = ROOT/'source'
    source_dir.mkdir(exist_ok=True)
    destination = source_dir/'epoch_025.pt'
    if not destination.exists():
        shutil.copyfile(SOURCE, destination.with_suffix('.tmp'))
        destination.with_suffix('.tmp').replace(destination)
    checksum = sha(SOURCE)
    if sha(destination) != checksum:
        raise RuntimeError('Copied checkpoint differs from source')
    (source_dir/'provenance.json').write_text(json.dumps(dict(original=str(SOURCE), sha256=checksum), indent=2))
    jobs = dict(source=str(SOURCE), source_sha256=checksum, training=[], evaluation=[])
    def save():
        record.write_text(json.dumps(jobs, indent=2))
    def submit(action, dependency=None, epochs=None):
        args = ['/usr/local/slurm/current/bin/sbatch', '--parsable', '--array=0-1%2']
        if dependency:
            args += ['--dependency=afterok:'+dependency, '--kill-on-invalid-dep=yes']
        if action == 'smoke':
            args += ['--time=00:20:00']
        args += ['scripts/film_finetune.sbatch', action]
        if epochs is not None:
            args += ['--epochs', str(epochs)]
        return subprocess.check_output(args, text=True).strip().split(';')[0]
    save()
    previous = submit('smoke')
    jobs['smoke'] = previous
    save()
    for epochs in (2, 4):
        previous = submit('train', previous, epochs)
        jobs['training'].append(dict(additional_epochs=epochs, job=previous)); save()
        # Evaluate immutable weights after each chunk. Keep total ablation use
        # at two GPUs by completing evaluation before the next training chunk.
        previous = submit('evaluate', previous, epochs)
        jobs['evaluation'].append(dict(additional_epochs=epochs, job=previous)); save()
    print(json.dumps(jobs, indent=2))

if __name__ == '__main__':
    main()
