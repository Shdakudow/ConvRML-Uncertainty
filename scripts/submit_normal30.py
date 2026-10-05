"""Submit prepared continuation once, persisting each job ID immediately."""
import json
from pathlib import Path
import subprocess
root=Path('results/normal_convrml_epoch30_20260924')
assert json.loads((root/'prepared.json').read_text())['start_epoch']==20
path=root/'jobs.json'
assert not path.exists(),'Jobs already recorded; inspect rather than submitting duplicates'
sb='/usr/local/slurm/current/bin/sbatch'
jobs={'prepare':'236088','training':[]};previous=None
for epoch in [22,24,26,28,30]:
    command=[sb,'--parsable','--job-name=rml-normal-'+str(epoch)]
    if previous:command.append('--dependency=afterok:'+previous)
    command+=['scripts/formal_quantile_chunk.sbatch','normal',str(epoch),'--root',str(root)]
    previous=subprocess.check_output(command,text=True).strip().split(';')[0]
    jobs['training'].append({'end_epoch':epoch,'job':previous})
    path.write_text(json.dumps(jobs,indent=2))
jobs['evaluation']=subprocess.check_output([sb,'--parsable','--dependency=afterok:'+previous,'scripts/normal30_evaluate.sbatch'],text=True).strip().split(';')[0]
path.write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs,indent=2))
