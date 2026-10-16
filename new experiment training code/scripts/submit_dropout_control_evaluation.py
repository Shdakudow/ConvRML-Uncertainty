"""Evaluate each five-epoch milestone after its two-epoch training job finishes."""
import json,subprocess
from pathlib import Path
root=Path('results/dropout_control_20261015');training=json.loads((root/'jobs.json').read_text())['jobs']
p=root/'evaluation_jobs.json';rows=json.loads(p.read_text()) if p.exists() else []
for camera in ['rml','diffuser']:
 previous=None
 for epoch in range(5,51,5):
  existing=next((r for r in rows if r['camera']==camera and r['epoch']==epoch),None)
  if existing:previous=existing['job'];continue
  parent=next(r['job'] for r in training if r['camera']==camera and r['stage']=='train' and r['epoch']==((epoch+1)//2)*2)
  deps=parent+(':'+previous if previous else '')
  args=['/usr/local/slurm/current/bin/sbatch','--parsable','--job-name=dropout-control-eval','--partition=gpu','--gres=gpu:1',
   '--constraint=(h100|h200|6000ada|a6000)&avx2','--prefer=h100|h200|6000ada','--exclude=monakhova-compute-01,nikola-compute-15,bhattacharjee-compute-02',
   '--cpus-per-task=2','--mem=32G','--time=04:00:00','--requeue','--dependency=afterok:'+deps,
   '--output=logs/dropout_control_eval_%j.out','--error=logs/dropout_control_eval_%j.err',
   '--wrap',f'$HOME/.conda/envs/convrml/bin/python -u scripts/evaluate_dropout_control.py --camera {camera} --epoch {epoch}']
  job=subprocess.check_output(args,text=True).strip().split(';')[0];rows.append(dict(camera=camera,epoch=epoch,job=job,dependency=deps));previous=job
  p.write_text(json.dumps(rows,indent=2))
print(json.dumps(dict(evaluations=len(rows),first=rows[0],last=rows[-1]),indent=2))
