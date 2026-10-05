"""Submit the compact comparison in two-epoch stages, gated by runtime smoke."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path('results/toy50_hour_20260926');SB='/usr/local/slurm/current/bin/sbatch'

def submit(args):return subprocess.check_output([SB,'--parsable',*args],text=True).strip().split(';')[0]

def main():
    p=argparse.ArgumentParser();p.add_argument('--training',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);record=ROOT/'jobs.json'
    if not a.training:
        assert not record.exists()
        prep=submit(['--job-name=toy50-prepare','--partition=monakhova','--constraint=avx2','--cpus-per-task=2','--mem=4G','--time=00:15:00','--output=logs/toy50_prepare_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/toy50_hour.py prepare'])
        jobs={'prepare':prep};record.write_text(json.dumps(jobs,indent=2))
        jobs['smoke']=submit(['--dependency=afterok:'+prep,'scripts/toy50_hour.sbatch','smoke']);record.write_text(json.dumps(jobs,indent=2))
    else:
        jobs=json.loads(record.read_text());assert 'training' not in jobs
        smoke=json.loads((ROOT/'smoke.json').read_text());assert smoke['passed']
        previous=None;jobs['training']=[]
        for epoch in range(2,51,2):
            previous=submit((['--dependency=afterok:'+previous] if previous else [])+['scripts/toy50_hour.sbatch','train','--epochs',str(epoch)])
            jobs['training'].append({'end_epoch':epoch,'job':previous});record.write_text(json.dumps(jobs,indent=2))
        jobs['evaluation']=submit(['--dependency=afterok:'+previous,'--time=00:20:00','scripts/toy50_hour.sbatch','evaluate']);record.write_text(json.dumps(jobs,indent=2))
    print(json.dumps(jobs,indent=2))

if __name__=='__main__':main()
