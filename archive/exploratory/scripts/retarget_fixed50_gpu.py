"""Retarget only pending fixed-backbone50 GPU jobs; retain dependencies."""
import json, subprocess, datetime
from pathlib import Path
root=Path('results/fixed_backbone50_20260927')
jobs=json.loads((root/'jobs.json').read_text())
slurm='/usr/local/slurm/current/bin/'
features='(h100|h200|6000ada)&avx2'
changes=[]
for job in [v['job'] for v in jobs['training']]+[jobs['inference']]:
    r=subprocess.run([slurm+'squeue','-h','-j',job,'-o','%T'],capture_output=True,text=True)
    if r.stdout.strip()!='PENDING':continue
    subprocess.run([slurm+'scontrol','update','JobId='+job,'Features='+features,'ExcNodeList=nikola-compute-15,bhattacharjee-compute-02'],check=True)
    check=subprocess.check_output([slurm+'scontrol','show','job',job,'-o'],text=True)
    assert 'Features='+features in check, check
    changes.append(job)
record={'timestamp':datetime.datetime.now().isoformat(),'features':features,'updated_jobs':changes,'running_jobs':'left untouched; resume from completed two-epoch chunk','reason':'User requested faster GPU; one H100 unallocated at inspection','excluded_mixed_blackwell_node':'Avoid unsupported GPU architecture in existing environment'}
(root/'gpu_retarget.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
