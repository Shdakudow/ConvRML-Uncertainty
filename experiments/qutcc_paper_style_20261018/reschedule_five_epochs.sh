#!/bin/bash
# Scheduling-only replacement. Preserve Python bytes and checkpoint source identity.
set -euo pipefail
cd "$(dirname "$0")"
/home/jy2228/.conda/envs/convrml/bin/python - <<'PY'
import json, subprocess, time
from pathlib import Path
root=Path.cwd(); slurm='/usr/local/slurm/current/bin/'
record=root/'five_epoch_jobs.json'
if record.exists(): raise SystemExit('Already rescheduled; refusing duplicate jobs')
old=json.loads((root/'jobs.json').read_text())
active=subprocess.check_output([slurm+'squeue','-u','jy2228','-h','-o','%i|%T'],text=True)
states=dict(line.split('|',1) for line in active.splitlines())
ids=[str(j) for j in old.values() if str(j) in states]
# Do not interrupt a running segment without reviewing its checkpoint first.
assert all(states[j]=='PENDING' for j in ids), 'An original job is running; inspect before rescheduling'
record.write_text(json.dumps({'status':'cancelling','cancelled':ids,'jobs':{}},indent=2))
if ids: subprocess.run([slurm+'scancel',*ids],check=True)
for _ in range(30):
    remaining=set(subprocess.check_output([slurm+'squeue','-u','jy2228','-h','-o','%i'],text=True).split()) & set(ids)
    if not remaining: break
    time.sleep(1)
else: raise RuntimeError('Cancellation not complete; refusing duplicate training')
meta={'status':'submitting','epochs_per_job':5,'qos':'normal','priority_note':'Only normal QoS is authorized for jy2228/monakhova','cancelled':ids,'jobs':{},'resume_epochs':{}}
def store(): record.write_text(json.dumps(meta,indent=2))
def submit(name,args,deps=()):
    cmd=[slurm+'sbatch','--parsable','--qos=normal','--nice=0','--job-name='+name]
    if deps: cmd+=['--dependency=afterok:'+':'.join(deps)]
    cmd+=['run.sbatch',*args]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    meta['jobs'][name]=job;store();print(name,job,flush=True);return job
for task,(label,camera,method) in enumerate([('rml-q','rml','qutcc'),('diff-q','diffuser','qutcc'),('raw-q','raw','qutcc'),('raw-im2im','raw','im2im')]):
    out=root/'results'/camera/method
    progress=json.loads((out/'progress.json').read_text())
    assert progress['stage']=='validated',progress
    current=progress['epoch'];meta['resume_epochs'][label]=current
    assert json.loads((out/'smoke.json').read_text())['passed']
    previous=None; milestones={}
    for epoch in range((current//5+1)*5,51,5):
        previous=submit(f'paper5-{label}-{epoch:02d}',['train.py','--task',str(task),'--target-epoch',str(epoch)],[previous] if previous else [])
        milestones[epoch]=previous
    evaluation=None
    for epoch in range(5,51,5):
        dest=out/'evaluation'/f'epoch_{epoch:03d}'
        if (dest/'report.json').exists() or (dest/'failure.json').exists():continue
        assert epoch<=current or epoch in milestones
        deps=([milestones[epoch]] if epoch in milestones else [])+([evaluation] if evaluation else [])
        evaluation=submit(f'paper5-eval-{label}-{epoch:02d}',['evaluate.py','--task',str(task),'--epoch',str(epoch)],deps)
meta['status']='submitted';store()
PY
