"""Independent training lanes: two-epoch jobs without global round barriers."""
import json,subprocess,argparse
from pathlib import Path
ROOT=Path('results/corrected_comparison_20261006')
ARMS=[f'ensemble_{i:02d}' for i in range(10)]+['conditioned','im2im','mc_005','mc_010','mc_020']
# Start with the four model types; spread each dataset and expensive arms across lanes.
ORDER=[10,25,11,26,12,27,0,15,1,16,2,17,13,28,14,29,3,18,4,19,5,20,6,21,7,22,8,23,9,24]

def plan(max_gpus=12):
    lanes=[ORDER[i::6] for i in range(6)]
    if max_gpus==12:lanes=[lane[:3] for lane in lanes]+[lane[3:] for lane in lanes]
    elif max_gpus!=6:raise ValueError('Supported limits are 6 or 12 GPUs')
    jobs=[]
    for lane,tasks in enumerate(lanes):
        for task in tasks:jobs.append(dict(lane=lane,stage='smoke',task=task,epoch=None))
        for epoch in range(2,51,2):
            for task in tasks:jobs.append(dict(lane=lane,stage='train',task=task,epoch=epoch))
    return lanes,jobs

def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--max-gpus',type=int,choices=[6,12],default=12);a=p.parse_args()
    lanes,jobs=plan(a.max_gpus)
    if a.dry_run:
        print(json.dumps(dict(lanes=lanes,training_jobs=sum(j['stage']=='train' for j in jobs),smoke_jobs=30,max_gpus=a.max_gpus)));return
    ROOT.mkdir(parents=True,exist_ok=True);Path('logs').mkdir(exist_ok=True)
    record=ROOT/'jobs.json'
    with record.open('x') as f:f.write('{}')
    config=json.loads(Path('configs/corrected_comparison_parameters.json').read_text())
    config.update(status='submitting',lanes=lanes,max_concurrent_gpus=a.max_gpus,qutcc_constraint='(h100|h200|6000ada|a6000)&avx2',jobs=[])
    tails=[None]*a.max_gpus
    def submit(stage,cli,lane=None,dependency=None,task=None,epoch=None):
        args=['/usr/local/slurm/current/bin/sbatch','--parsable']
        if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        if stage=='smoke':args+=['--time=00:20:00']
        args+=['scripts/corrected_comparison.sbatch',stage,*cli]
        job=subprocess.check_output(args,text=True).strip().split(';')[0]
        config['jobs'].append(dict(job=job,stage=stage,lane=lane,task=task,epoch=epoch,dependency=dependency))
        record.write_text(json.dumps(config,indent=2));return job
    try:
        # Submit in chronological rounds across lanes, not all of lane zero first.
        per_lane=[[j for j in jobs if j['lane']==i] for i in range(a.max_gpus)]
        for offset in range(max(map(len,per_lane))):
            for lane in range(a.max_gpus):
                if offset>=len(per_lane[lane]):continue
                j=per_lane[lane][offset];cli=['--task',str(j['task'])]
                if j['epoch'] is not None:cli+=['--epochs',str(j['epoch'])]
                tails[lane]=submit(j['stage'],cli,lane,tails[lane],j['task'],j['epoch'])
        barrier=':'.join(tails)
        tunes=[submit('tune',['--task',str(i)],dependency=barrier) for i in range(6)]
        selects=[submit('select',['--camera',camera],dependency=':'.join(tunes[3*i:3*i+3])) for i,camera in enumerate(('rml','diffuser'))]
        barrier=':'.join(selects)
        # Evaluation retains six-way concurrency after all twelve training lanes finish.
        for i in range(6):
            first=submit('evaluate',['--task',str(i)],dependency=barrier)
            submit('evaluate',['--task',str(i+6)],dependency=first)
        config['status']='submitted';record.write_text(json.dumps(config,indent=2))
        print(json.dumps(dict(jobs=len(config['jobs']),first_jobs=[j['job'] for j in config['jobs'][:6]],last_job=config['jobs'][-1]['job'],lanes=lanes)))
    except Exception:
        config['status']='submission_failed';record.write_text(json.dumps(config,indent=2));raise

if __name__=='__main__':main()
