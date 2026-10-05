import json,subprocess
from pathlib import Path
record=Path('reports/all_rcps_milestones_jobs.json')
if record.exists():raise SystemExit(record.read_text())
tasks=[]
for epoch in range(5,51,5):
 for worker in range(8):
  camera='rml' if worker<4 else 'diffuser'
  method=('ensemble','mc','fixed','conditioned')[worker%4]
  candidates=[Path(f'results/rcps_vs_empirical_epoch{epoch}')/camera/method/'comparison.json']
  if (camera,epoch) in [('rml',40),('diffuser',35)]:candidates.append(Path('results/rcps_vs_empirical_rml40_diffuser35_20261001')/camera/method/'comparison.json')
  if not any(p.exists() for p in candidates):tasks.append(dict(epoch=epoch,worker=worker))
Path('reports/all_rcps_tasks.json').write_text(json.dumps(tasks))
job=subprocess.check_output(['/usr/local/slurm/current/bin/sbatch','--parsable',f'--array=0-{len(tasks)-1}%6','scripts/all_rcps.sbatch'],text=True).strip()
record.write_text(json.dumps(dict(job=job,tasks=tasks),indent=2));print(record.read_text())
