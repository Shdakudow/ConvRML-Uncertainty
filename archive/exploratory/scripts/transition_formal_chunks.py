"""Watch saved epochs; replace old long allocations with dependent short jobs."""
import json,time,subprocess
from pathlib import Path
ROOT=Path('results/formal_rml_25k_epoch20')
SLURM='/usr/local/slurm/current/bin/'
def run(tool,*args):return subprocess.check_output([SLURM+tool,*args],text=True).strip()
def epoch(worker):
    p=ROOT/f'worker_{worker}/history.json'
    try:return json.loads(p.read_text())[-1]['epoch']
    except (FileNotFoundError,json.JSONDecodeError):return 8
def submit(script,dependency,*args):
    return run('sbatch','--parsable',f'--dependency={dependency}','--kill-on-invalid-dep=yes',f'scripts/{script}',*map(str,args)).split(';')[0]
def main():
    receipt=ROOT/'chunk_transition.json'
    if receipt.exists():raise FileExistsError('Transition already configured')
    plan={str(i):{'boundary':min(20,2*((epoch(i)//2)+1)),'jobs':[]} for i in range(6)}
    receipt.write_text(json.dumps(plan,indent=2))
    # Superseded aggregation depends on long allocations that will be cancelled.
    subprocess.run([SLURM+'scancel','673115'],check=False)
    pending=set(range(6));deadline=time.monotonic()+3*3600
    while pending:
        for i in sorted(pending):
            entry=plan[str(i)];current=epoch(i)
            if current<entry['boundary']:continue
            old=f'673045_{i}'
            # history.json is written only after latest.pt is atomically saved.
            # Polling can lose a few seconds of work started after the checkpoint.
            subprocess.run([SLURM+'scancel',old],check=False)
            dependency=f'afterany:{old}'
            for target in range(current+2,21,2):
                job=submit('formal_rml_chunk.sbatch',dependency,i,target)
                entry['jobs'].append({'job':job,'target_epoch':target});dependency=f'afterok:{job}'
            if current<20 and (20-current)%2:
                job=submit('formal_rml_chunk.sbatch',dependency,i,20)
                entry['jobs'].append({'job':job,'target_epoch':20});dependency=f'afterok:{job}'
            entry['inference_job']=submit('formal_rml_inference20.sbatch',dependency,i)
            entry['transition_epoch']=current;pending.remove(i)
            receipt.write_text(json.dumps(plan,indent=2));print(json.dumps({'worker':i,**entry}),flush=True)
        if time.monotonic()>deadline:raise TimeoutError('Some workers did not reach a saved boundary; inspect before retry')
        if pending:time.sleep(5)
    deps='afterok:'+':'.join(plan[str(i)]['inference_job'] for i in range(6))
    job=submit('formal_rml_aggregate20.sbatch',deps)
    plan['aggregation_job']=job;receipt.write_text(json.dumps(plan,indent=2));print('AGGREGATION',job,flush=True)
if __name__=='__main__':main()
