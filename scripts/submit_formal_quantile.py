import json,subprocess
from pathlib import Path
out=Path('results/formal_quantile_comparison');out.mkdir(parents=True,exist_ok=True)
if (out/'jobs.json').exists():raise FileExistsError('Job chain already exists')
jobs={}
def submit(script,args=(),dependency=None,smoke=False):
    cmd=['/usr/local/slurm/current/bin/sbatch','--parsable','--kill-on-invalid-dep=yes']
    if dependency:cmd.append('--dependency=afterok:'+str(dependency))
    if smoke:cmd.append('--time=00:15:00')
    job=subprocess.check_output(cmd+[f'scripts/{script}']+list(map(str,args)),text=True).strip().split(';')[0]
    return job
def save():(out/'jobs.json').write_text(json.dumps(jobs,indent=2))
for kind in ('normal','conditioned','fixed'):
    jobs[kind]=[]
    if kind=='fixed':previous=jobs['normal'][-1]['job']
    else:
        previous=submit('formal_quantile_chunk.sbatch',(kind,2,'--smoke'),smoke=True)
        jobs[kind].append({'stage':'smoke','job':previous});save()
    for end in range(2,21,2):
        previous=submit('formal_quantile_chunk.sbatch',(kind,end),previous)
        jobs[kind].append({'through_epoch':end,'job':previous});save()
    if kind!='normal':
        jobs[kind+'_inference']=submit('formal_quantile_infer.sbatch',(kind,),previous);save()
jobs['report']=submit('formal_quantile_report.sbatch',dependency=jobs['fixed_inference']+':'+jobs['conditioned_inference']);save()
print(json.dumps(jobs,indent=2))
