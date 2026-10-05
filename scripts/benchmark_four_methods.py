"""Same-device end-to-end GPU interval inference plus recorded training times."""
import argparse,gc,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.shamus_rml import ShamusRML
from models.formal_quantile import QuantileRML
from baselines.formal_rml import FormalRML
from baselines.deep_ensemble import DeepEnsembleMember
from baselines.mc_dropout import dropout_inference
OUT=Path('results/four_methods_runtime_20260925')
QR=Path('results/formal_quantile_comparison');BR=Path('results/formal_rml_25k_epoch20')

def training():
    OUT.mkdir(parents=True,exist_ok=True)
    records={}
    for key,path in [('normal',QR/'normal'),('fixed_head',QR/'fixed'),('conditioned',QR/'conditioned')]+[(f'worker_{i}',BR/f'worker_{i}') for i in range(6)]:
        history=json.loads((path/'history.json').read_text());assert history[-1]['epoch']==20
        records[key]={'epochs':20,'epoch_seconds':sum(row['seconds'] for row in history),'mean_seconds_per_epoch':np.mean([row['seconds'] for row in history])}
    (OUT/'training.json').write_text(json.dumps({'records':records,'definition':'Sum of logged completed epoch durations including validation; excludes queue time, startup, most checkpoint writes, and work lost to preemption. Independent workers can overlap; sums across members are GPU-work hours, not elapsed parallel duration.'},indent=2))
    print(json.dumps(records,indent=2))

def benchmark():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.manual_seed(20260925);torch.backends.cudnn.benchmark=False
    pairs=np.load('data/formal_rml_25k/pairs.npy',mmap_mode='r')
    y=torch.from_numpy(np.array(pairs[22500:22502,0])).float().cuda()/255
    factors=json.loads(Path('results/four_models_1000_alpha10/calibration.json').read_text())['factors']
    results={}
    for method in ['fixed','conditioned','ensemble','mc']:
        models=[]
        if method in ['fixed','conditioned']:
            model=QuantileRML(ShamusRML(),method)
            ck=torch.load(QR/method/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);del ck
            models=[model.cuda().eval()];del model
        else:
            for worker in (range(5) if method=='ensemble' else [5]):
                net=FormalRML('ensemble' if method=='ensemble' else 'mc')
                model=DeepEnsembleMember(net) if method=='ensemble' else net
                ck=torch.load(BR/f'worker_{worker}'/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);del ck
                models.append(model.cuda().eval());del model,net
        @torch.inference_mode()
        def run():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if method in ['fixed','conditioned']:
                    r,lo,hi=[v.float() for v in models[0].bounds(y)]
                    return r,r-factors[method]*(r-lo).clamp_min(.001),r+factors[method]*(hi-r).clamp_min(.001)
                if method=='ensemble':
                    values=[model(y) for model in models]
                    means=torch.stack([v[0].float() for v in values]);vars=torch.stack([v[1].float() for v in values])
                    r=means.mean(0);std=(vars.mean(0)+means.var(0,unbiased=False)).sqrt().clamp_min(1e-4)
                else:
                    with dropout_inference(models[0]):
                        r=None;m2=None
                        for i in range(30):
                            value=models[0](y).float()
                            if r is None:r=value.clone();m2=torch.zeros_like(value)
                            else:d=value-r;r=r+d/(i+1);m2=m2+d*(value-r)
                    std=(m2/30).sqrt().clamp_min(1e-4)
                return r,r-factors[method]*std,r+factors[method]*std
        for _ in range(3):run()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(12):
            torch.cuda.synchronize();start=time.perf_counter();output=run();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        assert all(torch.isfinite(v).all() for v in output)
        results[method]={'batch_size':2,'timed_batches':12,'mean_batch_seconds':float(np.mean(times)),
                         'median_batch_seconds':float(np.median(times)),'mean_ms_per_image':float(np.mean(times)*500),
                         'images_per_second':float(2/np.mean(times)), 'p10_batch_seconds':float(np.quantile(times,.1)),
                         'p90_batch_seconds':float(np.quantile(times,.9)),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
        print(method,json.dumps(results[method]),flush=True)
        del models,run,output;gc.collect();torch.cuda.empty_cache()
    payload={'gpu':torch.cuda.get_device_name(),'input':'RGB 256x256','precision':'bf16 autocast, float32 moment/interval arithmetic',
             'scope':'Warm GPU inference including reconstruction, interval heads, aggregation and calibrated bounds. Excludes checkpoint loading, data preprocessing, input transfer and saving. Same GPU and batch size; ensemble members sequential on this GPU; MC 30 stochastic passes.',
             'metrics':results}
    (OUT/'inference.json').write_text(json.dumps(payload,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['training','benchmark']);a=p.parse_args()
    training() if a.action=='training' else benchmark()
