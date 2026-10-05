"""Reserve two calibration slots and automatically restore the larger run."""
import argparse,json,subprocess
from pathlib import Path
BASE='/usr/local/slurm/current/bin/'
ROOT=Path('results/native_uncertainty_5k_20260926/calibrated95')

def throttle(limit):
    jobs=json.loads(Path('results/four_models_epoch50_20260925/jobs.json').read_text())
    candidates={v['array_job'] for v in jobs['training']}|{jobs['inference']}
    active=set(subprocess.check_output([BASE+'squeue','-u','jy2228','-h','-o','%A'],text=True).split())
    updated=[]
    for job in sorted(candidates & active):
        subprocess.run([BASE+'scontrol','update','JobId='+job,'ArrayTaskThrottle='+str(limit)],check=True);updated.append(job)
    print('GPU throttle',limit,updated,flush=True)
    return updated

def main():
    p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');a=p.parse_args()
    if a.restore:throttle(6);return
    ROOT.mkdir(parents=True,exist_ok=True);record=ROOT/'jobs.json'
    assert not record.exists(),'Existing calibration submission; inspect before retrying'
    def submit(args):return subprocess.check_output([BASE+'sbatch','--parsable',*args],text=True).strip().split(';')[0]
    changed=throttle(4)
    job=submit(['scripts/small7k_calibrate95.sbatch'])
    jobs={'inference':job,'temporarily_limited_jobs':changed};record.write_text(json.dumps(jobs,indent=2))
    cpu=['--partition=monakhova','--constraint=avx2','--cpus-per-task=4','--mem=12G','--time=01:00:00']
    jobs['restore_gpu_slots']=submit(['--dependency=afterany:'+job,'--job-name=rml-restore-slots',*cpu,'--output=logs/cal95_restore_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/submit_small7k_cal95.py --restore'])
    jobs['report']=submit(['--dependency=afterok:'+job,'--job-name=rml-cal95-report',*cpu,'--output=logs/cal95_report_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/small7k_calibrate95.py report'])
    record.write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs,indent=2))

if __name__=='__main__':main()
