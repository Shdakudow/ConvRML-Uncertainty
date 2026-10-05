"""Submit matched ten-epoch controls as two-epoch chains."""
import json,subprocess
from pathlib import Path
root=Path('results/controls30_20260924');root.mkdir(parents=True,exist_ok=True)
record=root/'jobs.json';assert not record.exists(),'Inspect existing jobs rather than duplicating'
sb='/usr/local/slurm/current/bin/sbatch'
def submit(args):return subprocess.check_output([sb,'--parsable',*args],text=True).strip().split(';')[0]
prep=submit(['--job-name=rml-controls-prepare','--partition=monakhova','--constraint=avx2','--cpus-per-task=2','--mem=12G','--time=00:10:00','--output=logs/controls_prepare_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/prepare_controls30.py'])
jobs={'prepare':prep,'arms':{}};record.write_text(json.dumps(jobs,indent=2))
for name,kind,options in [('mse_lr1e5','normal',['--lr','1e-5']),('joint_w002','conditioned',['--pinball-weight','0.02','--validation-mse','--isolated-q'])]:
    previous=prep;arm={'training':[]};jobs['arms'][name]=arm
    for epoch in [22,24,26,28,30]:
        previous=submit(['--dependency=afterok:'+previous,'--job-name='+name+'-'+str(epoch),'scripts/formal_quantile_chunk.sbatch',kind,str(epoch),'--root',str(root/name),*options])
        arm['training'].append({'end_epoch':epoch,'job':previous});record.write_text(json.dumps(jobs,indent=2))
    arm['evaluation']=submit(['--dependency=afterok:'+previous,'scripts/normal30_evaluate.sbatch','--root',str(root/name),'--kind',kind])
    record.write_text(json.dumps(jobs,indent=2))
print(json.dumps(jobs,indent=2))
