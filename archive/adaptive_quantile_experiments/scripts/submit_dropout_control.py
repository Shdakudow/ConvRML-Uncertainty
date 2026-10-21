"""Two independent smoke-gated p=0 training chains; two epochs per job."""
import json,subprocess
from pathlib import Path
root=Path('results/dropout_control_20261015');root.mkdir(parents=True,exist_ok=True)
assert json.loads((root/'matching_audit.json').read_text())['passed']
p=root/'jobs.json';record=json.loads(p.read_text()) if p.exists() else dict(jobs=[])
for camera in ['rml','diffuser']:
 previous=None
 for stage,epoch in [('smoke',None)]+[('train',e) for e in range(2,51,2)]:
  existing=next((r for r in record['jobs'] if r['camera']==camera and r['stage']==stage and r['epoch']==epoch),None)
  if existing:previous=existing['job'];continue
  args=['/usr/local/slurm/current/bin/sbatch','--parsable','--job-name=dropout0-'+camera]
  if previous:args+=['--dependency=afterok:'+previous]
  if stage=='smoke':args+=['--time=00:30:00']
  args+=['scripts/dropout_control.sbatch',stage,'--camera',camera,'--arm','mc_000']
  if epoch:args+=['--epochs',str(epoch)]
  job=subprocess.check_output(args,text=True).strip().split(';')[0]
  record['jobs'].append(dict(camera=camera,stage=stage,epoch=epoch,job=job,dependency=previous));previous=job
  tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(p)
print(json.dumps(dict(count=len(record['jobs']),first=record['jobs'][0],second_camera=record['jobs'][26]),indent=2))
