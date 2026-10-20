"""Submit independent frozen-test evaluations after checkpoint restoration."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert not (ROOT/'jobs.json').exists(),'Existing job record; refuse duplicate submission'
(ROOT/'logs').mkdir(exist_ok=True);jobs={}
for camera in ['rml','diffuser']:
 for method in ['im2im','ensemble5','mc_005','qutcc']:
  if method!='qutcc':
   arms=[f'ensemble_{i:02d}' for i in range(5)] if method=='ensemble5' else [method]
   for arm in arms:assert (ROOT/'weights'/camera/arm/'checkpoints/epoch_050.pt').exists()
  cmd=['/usr/local/slurm/current/bin/sbatch','--parsable','--qos=normal','--nice=0',f'--job-name=exp2metric-{camera}-{method}']
  if camera=='rml' and method=='qutcc':
   report=Path.home()/'ConvRML/experiments/qutcc_paper_style_20261018/results/rml/qutcc/evaluation/epoch_050/report.json'
   if not report.exists():cmd+=['--dependency=afterok:377111']
  job=subprocess.check_output(cmd+['run.sbatch',camera,method],cwd=ROOT,text=True).strip().split(';')[0];jobs[camera+'/'+method]=job;(ROOT/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');print(camera,method,job,flush=True)
