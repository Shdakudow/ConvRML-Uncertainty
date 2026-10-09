"""Rewire pending two-epoch jobs into twelve chains without cancelling running work."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path('results/corrected_comparison_20261006')
SLURM = '/usr/local/slurm/current/bin/'
ARMS = [f'ensemble_{i:02d}' for i in range(10)] + ['conditioned', 'im2im', 'mc_005', 'mc_010', 'mc_020']


def queue():
    raw = subprocess.check_output([SLURM+'squeue', '-u', 'jy2228', '-h', '-o', '%i|%T|%Q'], text=True)
    return {parts[0]: dict(state=parts[1], priority=int(parts[2]))
            for line in raw.splitlines() if len(parts := line.strip().split('|')) == 3}


def plan(config, live, epochs):
    original = config.get('original_six_lanes', config['lanes'])
    if len(original) != 6 or any(len(lane) != 5 for lane in original):
        raise ValueError('Expected the original six lanes with five networks each')
    lanes = [lane[:3] for lane in original] + [lane[3:] for lane in original]
    assert sorted(t for lane in lanes for t in lane) == list(range(30))
    changes = []
    tails = []
    for lane_id, tasks in enumerate(lanes):
        jobs = [j for j in config['jobs'] if j['stage'] == 'train' and j['task'] in tasks]
        jobs.sort(key=lambda j: (j['epoch'], tasks.index(j['task'])))
        active = []
        for job in jobs:
            if job['job'] in live:
                if live[job['job']]['state'] not in ('PENDING', 'RUNNING'):
                    raise ValueError(f"Wait for transitional job {job['job']}")
                active.append(job)
            elif epochs[job['task']] < job['epoch']:
                raise ValueError(f"Missing unfinished job {job['job']}; recover it first")
        running = [j for j in active if live[j['job']]['state'] == 'RUNNING']
        if len(running) > 1 or (running and running[0] != active[0]):
            raise ValueError('Running job must remain first in its new chain')
        previous = None
        for job in active:
            if live[job['job']]['state'] == 'PENDING':
                changes.append(dict(job=job['job'], dependency=previous, lane=lane_id, stage='train'))
            previous = job['job']
        if previous:
            tails.append(previous)
    barrier = ':'.join(tails) or None
    for job in config['jobs']:
        if job['stage'] == 'tune' and job['job'] in live:
            if live[job['job']]['state'] != 'PENDING':
                raise ValueError('Calibration/tuning must not overlap incomplete training')
            changes.append(dict(job=job['job'], dependency=barrier, lane=None, stage='tune'))
    return lanes, changes, tails


def save(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    record = ROOT/'jobs.json'
    config = json.loads(record.read_text())
    if config.get('max_concurrent_gpus') == 12:
        raise ValueError('Already expanded; inspect the saved audit before any further changes')
    epochs = {}
    for task in range(30):
        camera = ('rml', 'diffuser')[task//15]
        history = json.loads((ROOT/camera/ARMS[task%15]/'history.json').read_text())
        epochs[task] = history[-1]['epoch']
    live = queue()
    lanes, changes, tails = plan(config, live, epochs)
    print(json.dumps(dict(lanes=lanes, updates=len(changes), training_tails=tails, max_gpus=12)), flush=True)
    if not args.apply:
        return
    backup = ROOT/'jobs_before_12_gpus_20261009.json'
    if backup.exists():
        raise ValueError('A previous expansion attempt exists; inspect it before retrying')
    save(backup, config)
    owned = {j['job'] for j in config['jobs']}
    pending = sorted(j for j in owned & live.keys() if live[j]['state'] == 'PENDING')
    # Freeze dispatch while replacing dependencies. Running jobs are never held or cancelled.
    for offset in range(0, len(pending), 50):
        batch = pending[offset:offset+50]
        result = subprocess.run([SLURM+'scontrol', 'hold', ','.join(batch)], capture_output=True, text=True)
        if result.returncode:
            print(result.stderr, flush=True)
    live = queue()
    held = [j for j in pending if j in live and live[j]['state'] == 'PENDING']
    if any(live[j]['priority'] != 0 for j in held):
        raise RuntimeError('Could not hold every pending job; inspect and release held jobs before retrying')
    # A job may finish while being held; refresh completed epochs before rebuilding.
    for task in epochs:
        camera = ('rml', 'diffuser')[task//15]
        epochs[task] = json.loads((ROOT/camera/ARMS[task%15]/'history.json').read_text())[-1]['epoch']
    lanes, changes, tails = plan(config, live, epochs)
    audit = dict(status='updating', max_gpus=12, held_jobs=held, changes=[], lanes=lanes, tails=tails)
    audit_path = ROOT/'expansion_12_gpus_20261009.json'
    save(audit_path, audit)
    index = {j['job']: j for j in config['jobs']}
    for change in changes:
        dependency = 'afterok:'+change['dependency'] if change['dependency'] else ''
        subprocess.run([SLURM+'scontrol', 'update', 'JobId='+change['job'], 'Dependency='+dependency], check=True)
        index[change['job']].update(dependency=change['dependency'], lane=change['lane'])
        audit['changes'].append(change)
        save(audit_path, audit)
    config.update(original_six_lanes=config['lanes'], lanes=lanes, max_concurrent_gpus=12,
                  scheduling='12 independent training chains; existing six-way evaluation retained')
    save(record, config)
    audit['status'] = 'releasing'
    save(audit_path, audit)
    for offset in range(0, len(held), 50):
        subprocess.run([SLURM+'scontrol', 'release', ','.join(held[offset:offset+50])], check=True)
    audit['status'] = 'complete'
    save(audit_path, audit)
    print(json.dumps(dict(status='complete', updated=len(changes), max_gpus=12)), flush=True)


if __name__ == '__main__':
    main()
