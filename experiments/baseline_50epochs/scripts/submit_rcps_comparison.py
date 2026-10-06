import json
import subprocess
from pathlib import Path
ROOT=Path('results/rcps_vs_empirical_rml40_diffuser35_20261001')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    record=ROOT/'jobs.json'
    if record.exists():raise FileExistsError('Inspect existing submission record before retrying')
    jobs=dict(alpha=.1,delta=.05,calibration='Hoeffding-Bentkus image-level RCPS vs empirical',jobs=[])
    record.write_text(json.dumps(jobs,indent=2))
    def submit(workers,dependency=None):
        args=['/usr/local/slurm/current/bin/sbatch','--parsable','--array='+workers]
        if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        args+=['scripts/compare_rcps.sbatch']
        job=subprocess.check_output(args,text=True).strip().split(';')[0]
        jobs['jobs'].append(dict(job=job,workers=workers,dependency=dependency))
        record.write_text(json.dumps(jobs,indent=2))
    submit('0,1,3,4,5,7%2')
    for study,epoch,worker in [('optics_intervals_60k_5k_5k_20260928',40,2),('diffuser_intervals_60k_5k_5k_20260928',35,6)]:
        root=Path('results')/study
        ck=root/'heads'/f'backbone_{epoch:03d}'/'fixed/checkpoints/epoch_010.pt'
        dep=None
        if not ck.exists():
            source=json.loads((root/'jobs.json').read_text())
            dep=next(r['job'] for r in source['head_training'] if r['backbone_epoch']==epoch and r['head_epoch']==10)
        submit(str(worker),dep)
    print(json.dumps(jobs,indent=2))

if __name__=='__main__':main()
