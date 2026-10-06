"""Submit dependent jobs after the already submitted CPU preparation job."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('prepare_job',type=int);args=p.parse_args()
slurm='/usr/local/slurm/current/bin/sbatch'
out=Path('results/formal_rml_25k');out.mkdir(parents=True,exist_ok=True)
receipt=out/'jobs.json'
if receipt.exists():raise FileExistsError('Job chain already submitted')
jobs={'prepare':str(args.prepare_job)}
for name,script,dependency in [('smoke','formal_rml_smoke.sbatch','prepare'),('train','formal_rml_train.sbatch','smoke'),('aggregate','formal_rml_aggregate.sbatch','train')]:
    result=subprocess.check_output([slurm,'--parsable',f'--dependency=afterok:{jobs[dependency]}','--kill-on-invalid-dep=yes',f'scripts/{script}'],text=True).strip()
    jobs[name]=result.split(';')[0]
    receipt.write_text(json.dumps(jobs,indent=2))
    print(name,jobs[name],flush=True)
