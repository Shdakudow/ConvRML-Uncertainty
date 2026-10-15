"""Submit independent, resumable two-epoch chains for both cameras after smoke gates."""
import json,subprocess
from pathlib import Path
ROOT=Path('results/adaptive_q_coverage90_scratch50_20261014')
PROTOCOL='joint_adaptive_q_coverage90_scratch50_v1'

def main():
 path=ROOT/'jobs.json';record=json.loads(path.read_text()) if path.exists() else dict(protocol=PROTOCOL,jobs=[])
 if record['protocol']!=PROTOCOL:raise ValueError('Submission protocol mismatch')
 for camera in ['rml','diffuser']:
  smoke=json.loads((ROOT/camera/'smoke.json').read_text())
  if not smoke['passed'] or smoke['protocol']!=PROTOCOL or smoke['initialization']!='random_no_pretrained':raise ValueError('Smoke gate failed')
  previous=None
  for epoch in range(2,51,2):
   existing=next((r for r in record['jobs'] if r['camera']==camera and r['target_epoch']==epoch),None)
   if existing:previous=existing['job'];continue
   args=['/usr/local/slurm/current/bin/sbatch','--parsable','--job-name=adaptive50-'+camera]
   if previous:args+=['--dependency=afterok:'+previous]
   args+=['scripts/constrained_adaptive50.sbatch','train','--camera',camera,'--target-epoch',str(epoch)]
   job=subprocess.check_output(args,text=True).strip().split(';')[0]
   record['jobs'].append(dict(camera=camera,target_epoch=epoch,job=job,dependency=previous));previous=job
   temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)
 record['status']='submitted';path.write_text(json.dumps(record,indent=2))
 print(json.dumps(dict(jobs=len(record['jobs']),first_jobs=[r for r in record['jobs'] if r['target_epoch']==2],last_jobs=[r for r in record['jobs'] if r['target_epoch']==50]),indent=2))
if __name__=='__main__':main()
