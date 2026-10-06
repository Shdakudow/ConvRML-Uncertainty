"""Queue a bounded six-GPU experiment with smoke and two-epoch dependencies."""
import json,subprocess
from pathlib import Path
ROOT=Path('results/selective_film50_both_cameras_20261005')
ROOT.mkdir(parents=True,exist_ok=True);record=ROOT/'jobs.json'
if record.exists():raise SystemExit('Already submitted: '+str(record))
config=dict(initialization='Fresh QUTCC; resume pilot extra ensemble and MC; preserve original im2im and ensemble members 0-4',backbone='ConvNeXtV2-Base FPN',film_sites=48,conditioning='FiLM only in QUTCC; no FiLM in ensemble or MC dropout',cameras=['rml','diffuser'],train_count=60000,val_count=5000,test_count=5000,calibration_count=2048,epochs=50,fixed_head_extra_epochs=0,ensemble_members=10,dropout_probabilities=[.05,.1,.2],validation_passes=[30,60,100],selected_inference_passes=100,coverage_alpha=.1,rcps_delta=.05,selection='Full validation SSIM at 100 passes; per-camera selected p locked before test',max_concurrent_gpus=6,checkpoints=list(range(5,51,5)),epochs_per_job=2,jobs=[])
record.write_text(json.dumps(config,indent=2))
def submit(stage,array,dependency=None,epochs=None):
 args=['/usr/local/slurm/current/bin/sbatch','--parsable','--array='+array]
 if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
 if 'smoke' in stage:args+=['--time=00:20:00']
 args+=['scripts/selective_film50_study.sbatch',stage]
 if epochs is not None:args+=['--round',str(epochs)]
 job=subprocess.check_output(args,text=True).strip().split(';')[0]
 config['jobs'].append(dict(stage=stage,array=array,round_number=epochs,dependency=dependency,job=job));record.write_text(json.dumps(config,indent=2));return job
last=submit('smoke','0-17%6')
for round_number in range(1,26):last=submit('train','0-17%6',last,round_number)
last=submit('tune','0-5%6',last)
last=submit('select','0-1%2',last)
last=submit('evaluate','0-11%6',last)
print(json.dumps(config,indent=2))
