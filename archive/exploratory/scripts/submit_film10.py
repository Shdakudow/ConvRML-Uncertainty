"""Queue a bounded six-GPU experiment with smoke and two-epoch dependencies."""
import json,subprocess
from pathlib import Path
ROOT=Path('results/film10_both_cameras_20261004')
ROOT.mkdir(parents=True,exist_ok=True);record=ROOT/'jobs.json'
if record.exists():raise SystemExit('Already submitted: '+str(record))
config=dict(initialization='fresh',cameras=['rml','diffuser'],train_count=60000,val_count=5000,test_count=5000,calibration_count=2048,epochs=10,fixed_head_extra_epochs=10,ensemble_members=10,dropout_probabilities=[.05,.1,.2],validation_passes=[30,60,100],selected_inference_passes=100,coverage_alpha=.1,rcps_delta=.05,selection='Full validation SSIM at 100 passes; per-camera selected p locked before test',max_concurrent_gpus=6,checkpoints=[5,10],epochs_per_job=2,jobs=[])
record.write_text(json.dumps(config,indent=2))
def submit(stage,array,dependency=None,epochs=None):
 args=['/usr/local/slurm/current/bin/sbatch','--parsable','--array='+array]
 if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
 if 'smoke' in stage:args+=['--time=00:20:00']
 args+=['scripts/film10_study.sbatch',stage]
 if epochs is not None:args+=['--epochs',str(epochs)]
 job=subprocess.check_output(args,text=True).strip().split(';')[0]
 config['jobs'].append(dict(stage=stage,array=array,epochs=epochs,dependency=dependency,job=job));record.write_text(json.dumps(config,indent=2));return job
last=submit('smoke','0-31%6')
for epoch in (2,4,6,8,10):last=submit('train','0-31%6',last,epoch)
last=submit('head-smoke','0-1%2',last)
for epoch in (2,4,6,8,10):last=submit('head-train','0-1%2',last,epoch)
last=submit('tune','0-5%6',last)
last=submit('select','0-1%2',last)
last=submit('evaluate','0-13%6',last)
print(json.dumps(config,indent=2))
