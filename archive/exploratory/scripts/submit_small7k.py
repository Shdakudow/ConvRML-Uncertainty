"""Submit guarded two-epoch chunks, plus inference and native uncertainty scores."""
import argparse,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--after');p.add_argument('--epochs',type=int,default=20);a=p.parse_args()
    assert a.epochs>0 and a.epochs%2==0
    root=Path('results/native_uncertainty_5k_20260926');root.mkdir(parents=True,exist_ok=True)
    record=root/'jobs.json';assert not record.exists(),'Inspect existing jobs before resubmitting'
    protocol={'epochs_per_stage':a.epochs,'split':[5000,1000,1000],'max_concurrent_gpus':6,'initialization':'from scratch; no pretrained weights or historical checkpoints',
              'training':'AdamW lr1e-4 wd1e-4 batch2 bf16 gradclip1; identical seeded epoch shuffles; paired model initialization; two-epoch resumable jobs',
              'methods':{'ensemble_nll':'5 independent Gaussian members, Gaussian NLL',
                         'ensemble_crps':'5 independent Gaussian members, per-member Gaussian CRPS; not joint mixture-CRPS training',
                         'mc_mse':'decoder Dropout2d .1, MSE',
                         'mc_crps':'same dropout network, unbiased sample CRPS using 4 draws per batch; native sample distribution',
                         'q_conditioned':'original MSE + uniform-q pinball, weight1; already half the quantile integral; no duplicate rescaled-loss run',
                         'fixed_head':'normal MSE reconstruction stage followed by frozen backbone + two endpoint pinball head stage; interval reference only'},
              'evaluation':'Best own-validation-loss checkpoint. 1000 shared test images; all-pixel PSNR/SSIM; 1024 shared sampled pixel-channel values/image for distribution scores. Native95% intervals, no post-calibration. Ensemble full Gaussian-mixture CRPS; MC empirical30-draw CRPS; q sorted31-midpoint approximation, raw quantile loss/crossing separately. Fixed endpoints get interval score, no invented CRPS.',
              'fairness_limits':['Epochs/data/paired seeds matched, computation is not: ensemble has five networks; MC CRPS has four training draws; fixed head has a second stage. Report compute times.',
                                 'One paired model seed; ensemble members are not replicated full ensembles. Existing dataset reused, near-duplicate scenes not certified.']}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    sb='/usr/local/slurm/current/bin/sbatch'
    def submit(args):return subprocess.check_output([sb,'--parsable',*args],text=True).strip().split(';')[0]
    prep=submit(['--job-name=rml-5k-prepare','--partition=monakhova','--constraint=avx2','--cpus-per-task=2','--mem=4G','--time=00:20:00','--output=logs/small7k_prepare_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/small7k_train.py prepare'])
    jobs={'prepare':prep,'training':[],'fixed_head':[]};record.write_text(json.dumps(jobs,indent=2))
    # Two short smoke workers can use the two idle slots while the four
    # remaining epoch-32 workers finish; full training waits for that stage.
    previous=submit(['--dependency=afterok:'+prep,'--array=0,5,6,7,9,14%2','scripts/small7k.sbatch','smoke'])
    jobs['smoke']=previous;record.write_text(json.dumps(jobs,indent=2))
    for epoch in range(2,a.epochs+1,2):
        dependency='afterok:'+previous+(':'+a.after if epoch==2 and a.after else '')
        previous=submit(['--dependency='+dependency,'--array=0-7,9-14%6','scripts/small7k.sbatch','train','--epochs',str(epoch)])
        jobs['training'].append({'end_epoch':epoch,'array_job':previous});record.write_text(json.dumps(jobs,indent=2))
    for epoch in range(2,a.epochs+1,2):
        previous=submit(['--dependency=afterok:'+previous,'--array=8','scripts/small7k.sbatch','train','--epochs',str(epoch)])
        jobs['fixed_head'].append({'end_epoch':epoch,'array_job':previous});record.write_text(json.dumps(jobs,indent=2))
    previous=submit(['--dependency=afterok:'+previous,'--array=0-5,7-14%6','scripts/small7k.sbatch','infer'])
    jobs['inference']=previous;record.write_text(json.dumps(jobs,indent=2))
    jobs['report']=submit(['--dependency=afterok:'+previous,'--job-name=rml-5k-report','--partition=monakhova','--constraint=avx2','--cpus-per-task=4','--mem=12G','--time=02:00:00','--output=logs/small7k_report_%j.out','--wrap','cd ~/ConvRML && ~/.conda/envs/convrml/bin/python scripts/small7k_eval.py report'])
    record.write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs,indent=2))

if __name__=='__main__':main()
