"""Preparation, smoke gates and 25 dependent two-epoch training arrays."""
import json
import subprocess
from pathlib import Path
from diffuser_interval_study import ROOT


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    record = ROOT / 'jobs.json'
    if record.exists():
        raise FileExistsError('Study already submitted; inspect jobs before any retry')
    Path('logs').mkdir(exist_ok=True)
    jobs = {'training': [], 'head_training': [], 'evaluation': [],
            'settings': {'camera': 'DiffuserCam', 'train': 60000, 'val': 5000, 'test': 5000, 'calibration': 2048,
                         'epochs': 50, 'save_every': 5, 'alpha': .10, 'max_parallel_training_gpus': 6}}
    def submit(args):
        job = subprocess.check_output(['/usr/local/slurm/current/bin/sbatch', '--parsable', *args], text=True).strip().split(';')[0]
        return job
    def save():
        record.write_text(json.dumps(jobs, indent=2))
    # Create a reservation record first so a partial submission cannot be duplicated.
    save()
    prep = submit(['--job-name=diffuser60k-prep', '--partition=monakhova', '--constraint=avx2',
                   '--cpus-per-task=2', '--mem=8G', '--time=04:00:00',
                   '--output=logs/diffuser60k_prep_%j.out', '--wrap',
                   'cd ~/ConvRML && ~/.conda/envs/convrml/bin/python -u scripts/prepare_diffuser_intervals.py'])
    jobs['prepare'] = prep
    save()
    previous = submit(['--dependency=afterok:'+prep, '--kill-on-invalid-dep=yes', '--array=0,5,6,7%4',
                       'scripts/diffuser_interval_study.sbatch', 'smoke'])
    jobs['smoke'] = previous
    save()
    for epoch in range(2, 51, 2):
        previous = submit(['--dependency=afterok:'+previous, '--kill-on-invalid-dep=yes', '--array=0-7%6',
                           'scripts/diffuser_interval_study.sbatch', 'train', '--epochs', str(epoch)])
        jobs['training'].append({'end_epoch': epoch, 'job': previous})
        save()
    training_jobs={j['end_epoch']:j['job'] for j in jobs['training']}
    eval_jobs=[]
    for backbone_epoch in range(5,51,5):
        previous=training_jobs[backbone_epoch if backbone_epoch%2==0 else backbone_epoch+1]
        for head_epoch in range(2,11,2):
            previous=submit(['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes','--array=0',
                             'scripts/diffuser_interval_study.sbatch','head','--backbone-epoch',str(backbone_epoch),
                             '--epochs',str(head_epoch)])
            jobs['head_training'].append({'backbone_epoch':backbone_epoch,'head_epoch':head_epoch,'job':previous})
            save()
        evaluation=submit(['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes','--array=0-3%2',
                           'scripts/diffuser_interval_study.sbatch','evaluate','--backbone-epoch',str(backbone_epoch)])
        jobs['evaluation'].append({'backbone_epoch':backbone_epoch,'job':evaluation})
        eval_jobs.append(evaluation)
        save()
    jobs['plot']=submit(['--dependency=afterok:'+':'.join(eval_jobs),'--kill-on-invalid-dep=yes',
                         '--job-name=diffuser60k-plots','--partition=monakhova','--constraint=avx2',
                         '--cpus-per-task=2','--mem=4G','--time=00:10:00','--output=logs/diffuser60k_plot_%j.out',
                         '--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/diffuser_interval_study.py plot'])
    save()
    print(json.dumps(jobs, indent=2))


if __name__ == '__main__':
    main()
