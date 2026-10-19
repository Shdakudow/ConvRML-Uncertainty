"""Approve-gated adaptation on reviewed data: GPU smoke, 1–5, 6–10, evaluations."""
import json,subprocess
from pathlib import Path
from data import audit
ROOT=Path(__file__).resolve().parent
SB='/usr/local/slurm/current/bin/sbatch'
if not (ROOT/'TRAINING_APPROVED').exists():raise SystemExit('Awaiting explicit approval')
if (ROOT/'jobs.json').exists():raise SystemExit('Existing job record: refusing duplicates')
for test in ['test_corruptions.py','test_materialized_data.py']:
    subprocess.run(['/home/jy2228/.conda/envs/convrml/bin/python',test],check=True,cwd=ROOT)
print('Verified reviewed data split:',audit('raw'),flush=True)
(ROOT/'logs').mkdir(exist_ok=True)
meta={'protocol':'raw_measurement_corruptions_pilot_20261018_v1','parent_epoch':30,'additional_epochs':10,'job_chunk_epochs':5,'dataset':'exact fixed reviewed dataset/manifest.json','jobs':{}}
def submit(name,args,deps=(),smoke=False):
    cmd=[SB,'--parsable','--qos=normal','--nice=0','--job-name='+name]
    if smoke:cmd+=['--time=00:20:00']
    if deps:cmd+=['--dependency=afterok:'+':'.join(deps)]
    job=subprocess.check_output(cmd+['run.sbatch',*args],cwd=ROOT,text=True).strip().split(';')[0]
    meta['jobs'][name]=job
    tmp=ROOT/'jobs.tmp';tmp.write_text(json.dumps(meta,indent=2));tmp.replace(ROOT/'jobs.json')
    print(name,job,flush=True);return job
for task,method in enumerate(['qutcc','im2im']):
    smoke=submit('corrupt-smoke-'+method,['train.py','--task',str(task),'--smoke'],smoke=True)
    first=submit('corrupt-'+method+'-05',['train.py','--task',str(task),'--target-epoch','5'],[smoke])
    second=submit('corrupt-'+method+'-10',['train.py','--task',str(task),'--target-epoch','10'],[first])
    evaluation=submit('corrupt-eval-'+method+'-05',['evaluate.py','--task',str(task),'--epoch','5'],[first])
    submit('corrupt-eval-'+method+'-10',['evaluate.py','--task',str(task),'--epoch','10'],[second,evaluation])
