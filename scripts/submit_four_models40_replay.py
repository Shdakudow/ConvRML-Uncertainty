import json,subprocess
from pathlib import Path
root=Path('results/four_models_epoch40_replay_20260927');root.mkdir(parents=True,exist_ok=True)
record=root/'jobs.json';assert not record.exists(),'Existing jobs recorded; inspect before resubmitting'
sb='/usr/local/slurm/current/bin/sbatch'
def submit(args):return subprocess.check_output([sb,'--parsable',*args],text=True).strip().split(';')[0]
previous=submit(['--job-name=rml-four40replay-prep','--partition=monakhova','--constraint=avx2','--cpus-per-task=2','--mem=8G','--time=00:20:00','--output=logs/four40replay_prepare_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/four_models40_replay.py prepare'])
jobs={'prepare':previous,'training':[]};record.write_text(json.dumps(jobs,indent=2))
for epoch in range(32,41,2):
    previous=submit(['--dependency=afterok:'+previous,'scripts/four_models40_replay.sbatch','train','--epochs',str(epoch)])
    jobs['training'].append({'end_epoch':epoch,'array_job':previous});record.write_text(json.dumps(jobs,indent=2))
previous=submit(['--dependency=afterok:'+previous,'scripts/four_models40_replay.sbatch','infer']);jobs['inference']=previous;record.write_text(json.dumps(jobs,indent=2))
jobs['report']=submit(['--dependency=afterok:'+previous,'--job-name=rml-four40replay-report','--partition=monakhova','--constraint=avx2','--cpus-per-task=4','--mem=16G','--time=01:00:00','--output=logs/four40replay_report_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/four_models40_replay.py report'])
record.write_text(json.dumps(jobs,indent=2))
jobs['visuals']=submit(['--dependency=afterok:'+jobs['report'],'--job-name=rml-four40replay-visuals','--partition=monakhova','--constraint=avx2','--cpus-per-task=2','--mem=8G','--time=00:20:00','--output=logs/four40replay_visuals_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/render_uncertainty_comparison.py --input results/four_models_epoch40_replay_20260927/examples/examples.npz --out results/four_models_epoch40_replay_20260927/visuals'])
record.write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs,indent=2))
