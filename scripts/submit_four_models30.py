import json,subprocess
from pathlib import Path
root=Path('results/four_models_epoch30_20260925');root.mkdir(parents=True,exist_ok=True)
record=root/'jobs.json';assert not record.exists(),'Existing jobs recorded; inspect before resubmitting'
sb='/usr/local/slurm/current/bin/sbatch'
def submit(args):return subprocess.check_output([sb,'--parsable',*args],text=True).strip().split(';')[0]
previous=submit(['--job-name=rml-four30-prep','--partition=monakhova','--constraint=avx2','--cpus-per-task=2','--mem=8G','--time=00:20:00','--output=logs/four30_prepare_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/four_models30.py prepare'])
jobs={'prepare':previous,'training':[]};record.write_text(json.dumps(jobs,indent=2))
for epoch in [22,24,26,28,30]:
    previous=submit(['--dependency=afterok:'+previous,'scripts/four_models30.sbatch','train','--epochs',str(epoch)])
    jobs['training'].append({'end_epoch':epoch,'array_job':previous});record.write_text(json.dumps(jobs,indent=2))
previous=submit(['--dependency=afterok:'+previous,'scripts/four_models30.sbatch','infer']);jobs['inference']=previous;record.write_text(json.dumps(jobs,indent=2))
jobs['report']=submit(['--dependency=afterok:'+previous,'--job-name=rml-four30-report','--partition=monakhova','--constraint=avx2','--cpus-per-task=4','--mem=16G','--time=01:00:00','--output=logs/four30_report_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/four_models30.py report'])
record.write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs,indent=2))
