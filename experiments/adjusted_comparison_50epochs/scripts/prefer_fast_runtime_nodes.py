"""Prefer faster GPUs/hosts for this study's pending jobs, retaining fallback eligibility."""
import json
import subprocess
from pathlib import Path

root = Path('results/corrected_comparison_20261006')
slurm = '/usr/local/slurm/current/bin/'
config = json.loads((root/'jobs.json').read_text())
owned = {job['job'] for job in config['jobs']}
pending = set(subprocess.check_output([slurm+'squeue', '-u', 'jy2228', '-t', 'PENDING', '-h', '-o', '%i'], text=True).split())
preference = 'h100|h200|6000ada|amd'
audit = []
for job in sorted(owned & pending, key=int):
    command = [slurm+'scontrol', 'update', 'JobId='+job, 'Prefer='+preference]
    result = subprocess.run(command, text=True, capture_output=True)
    audit.append(dict(job=job, returncode=result.returncode, message=result.stderr.strip()))
    if result.returncode and 'no longer pending' not in result.stderr:
        raise RuntimeError(result.stderr)
(root/'faster_node_preferences_20261009.json').write_text(json.dumps(dict(preference=preference,jobs=audit), indent=2))
print(json.dumps(dict(updated=sum(row['returncode']==0 for row in audit),total=len(audit),preference=preference)))
