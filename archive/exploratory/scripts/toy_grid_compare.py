"""Frozen toy backbone: five quantiles, matched initialization and updates."""
import sys,json,time,copy,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from models.convnext import QuantileConvRecon
from models.quantile_grid import GridHead,fresh_head,pinball,LEVELS
from toy_dataset import FourierCropDataset

OUT=Path('results/toy_grid_compare')
def digest(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-count',type=int,default=1024)
    parser.add_argument('--val-count',type=int,default=256)
    parser.add_argument('--test-count',type=int,default=512)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--offset',type=int,default=28000)
    parser.add_argument('--seed',type=int,default=20260921)
    parser.add_argument('--out',default='results/toy_grid_compare')
    args=parser.parse_args()
    if min(args.train_count,args.val_count,args.test_count,args.epochs)<=0:parser.error('Counts must be positive')
    global OUT
    OUT=Path(args.out)
    if (OUT/'latest.pt').exists() or (OUT/'results.json').exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {OUT}')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'config.json').write_text(json.dumps(vars(args),indent=2))
    ck=torch.load('checkpoints/motion_uniform_gn/best_model.pth',map_location='cpu',weights_only=False)
    c=OmegaConf.create(ck['config'])
    base=QuantileConvRecon(3,64,64,model_size=c.model.size,drop_path_rate=c.model.drop_path_rate,layer_scale_init_value=c.model.layer_scale_init_value,decoder_channels=c.model.decoder_channels,quantile_normalization='group',quantile_embedding_dim=c.quantile.embedding_dim,quantile_condition_channels=c.quantile.condition_channels)
    conditioner=copy.deepcopy(base.quantile_conditioner)
    base.load_state_dict(ck['model_state_dict'],strict=True);base.eval().requires_grad_(False).cuda()
    before=digest(base)
    template=fresh_head(base.decoder.FNN1007).cpu()
    conditional=copy.deepcopy(template).cuda();conditioner.cuda()
    grid=GridHead(template).cuda()
    datasets={role:FourierCropDataset('/share/monakhova/shamus_data/parallel_100k',offset,count,64,48,seed,'motion') for role,offset,count,seed in [('train',args.offset,args.train_count,args.seed+30000000),('val',args.offset+args.train_count,args.val_count,args.seed+40000000),('test',args.offset+args.train_count+args.val_count,args.test_count,args.seed+50000000)]}
    old=FourierCropDataset('/share/monakhova/shamus_data/parallel_100k',int(c.toy.source_offset),sum(int(c.dataset[k]) for k in ('train_count','val_count','test_count')),64,48,0,'motion')
    groups=[set(old.files)]+[set(d.files) for d in datasets.values()]
    assert all(not groups[i]&groups[j] for i in range(len(groups)) for j in range(i))
    (OUT/'manifest.json').write_text(json.dumps({k:d.files for k,d in datasets.items()},indent=2))
    # Cache small input images, not the very large full-resolution fused features.
    data={}
    for role,d in datasets.items():
        samples=list(DataLoader(d,batch_size=32,num_workers=2))
        data[role]=(torch.cat([b[0] for b in samples]),torch.cat([b[1] for b in samples]))
    optimizers={'conditioned':torch.optim.AdamW(list(conditional.parameters())+list(conditioner.parameters()),lr=3e-4,weight_decay=.001),'grid':torch.optim.AdamW(grid.parameters(),lr=3e-4,weight_decay=.001)}
    def predict(method,f):
        if method=='grid':return grid(f)
        return torch.stack([conditional(conditioner(f,torch.full((len(f),),q,device='cuda'))) for q in LEVELS],1)
    def validate(method):
        total=0.
        with torch.inference_mode():
            for y,x in zip(data['val'][0].split(8),data['val'][1].split(8)):
                f=base._encode_fused(y.cuda())
                total+=pinball(predict(method,f),x.cuda()).item()*len(y)
        return total/len(data['val'][0])
    with torch.no_grad():
        f=base._encode_fused(data['train'][0][:1].cuda())
        initial_error=float((predict('grid',f)-predict('conditioned',f)).abs().max())
        assert initial_error<2e-5,initial_error
    del f
    history=[];times={k:0. for k in optimizers};peak={k:0 for k in optimizers}
    # Equal epochs/updates, rather than equal wall time. Alternate execution order.
    for epoch in range(args.epochs):
        order=torch.randperm(len(data['train'][0]),generator=torch.Generator().manual_seed(100+epoch))
        for method in (['conditioned','grid'] if epoch%2==0 else ['grid','conditioned']):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();losses=[]
            for indices in order.split(8):
                y,x=[v[indices].cuda() for v in data['train']]
                with torch.no_grad():f=base._encode_fused(y)
                opt=optimizers[method];opt.zero_grad(set_to_none=True)
                if method=='conditioned':
                    loss_value=0.
                    # Accumulate exactly the five-level mean gradient without retaining five graphs.
                    for q in LEVELS:
                        pred=conditional(conditioner(f,torch.full((len(f),),q,device='cuda')))
                        loss=pinball(pred[:,None],x,(q,))/5;loss.backward();loss_value+=loss.item()
                else:
                    loss=pinball(grid(f),x);loss.backward();loss_value=loss.item()
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.)
                opt.step();losses.append(loss_value)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start;times[method]+=elapsed
            peak[method]=max(peak[method],torch.cuda.max_memory_allocated())
            record={'epoch':epoch+1,'method':method,'loss':float(np.mean(losses)),'seconds':elapsed,'validation_pinball':validate(method)}
            history.append(record);print(json.dumps(record),flush=True)
            torch.save({'grid':grid.state_dict(),'conditioned':conditional.state_dict(),'conditioner':conditioner.state_dict(),'history':history,'optimizers':{k:v.state_dict() for k,v in optimizers.items()},'config':vars(args),'rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state()},OUT/'latest.tmp')
            (OUT/'latest.tmp').replace(OUT/'latest.pt')
            (OUT/'history.json').write_text(json.dumps(history,indent=2))
    conditional.eval();conditioner.eval();grid.eval()
    report={'levels':LEVELS,'epochs':args.epochs,'train_count':args.train_count,'validation_count':args.val_count,'test_count':args.test_count,'config':vars(args),'initial_prediction_max_error':initial_error,'history':history,'training_seconds':times,'peak_allocated_bytes_both_heads_resident':peak,'parameters':{'grid':sum(p.numel() for p in grid.parameters()),'conditioned':sum(p.numel() for m in (conditioner,conditional) for p in m.parameters())},'test':{},'notes':['Single seed; empirical evaluation, no calibration or dropout.','Frozen previously trained toy backbone. Fresh heads copied from reconstruction decoder with GroupNorm.','Equal updates and examples; five levels per update; no checkpoint selection.','Head timing uses cached features; end-to-end timing includes backbone and reconstruction.']}
    with torch.inference_mode():
        for method in optimizers:
            predictions=[];recons=[]
            for y in data['test'][0].split(8):
                f=base._encode_fused(y.cuda());predictions.append(predict(method,f).cpu());recons.append(base.decoder.FNN1007(f).cpu())
            p=torch.cat(predictions);r=torch.cat(recons);x=data['test'][1];lo,hi=p[:,0],p[:,-1]
            report['test'][method]={'pinball':pinball(p,x).item(),'cdf_by_level':(x[:,None]<=p).float().mean((0,2,3,4)).tolist(),'coverage':((lo<=x)&(x<=hi)).float().mean().item(),'width':(hi-lo).clamp_min(0).mean().item(),'adjacent_crossing':(p[:,:-1]>p[:,1:]).float().mean().item(),'psnr':float(-10*torch.log10(((r-x)**2).mean()))}
            report['test'][method].update(strict_cdf_by_level=(x[:,None]<p).float().mean((0,2,3,4)).tolist(),lower_miss=(x<lo).float().mean().item(),upper_miss=(x>hi).float().mean().item())
            np.savez_compressed(OUT/f'{method}_examples.npz',y=data['test'][0][:10].numpy(),x=x[:10].numpy(),r=r[:10].numpy(),quantiles=p[:10].numpy())
            y=data['test'][0][:1].cuda();f=base._encode_fused(y)
            for scope in ('head','end_to_end'):
                def run():
                    if scope=='head':return predict(method,f)
                    z=base._encode_fused(y);return base.decoder.FNN1007(z),predict(method,z)
                for _ in range(10):run()
                timings=[]
                for _ in range(30):
                    torch.cuda.synchronize();t=time.perf_counter();run();torch.cuda.synchronize();timings.append((time.perf_counter()-t)*1000)
                report['test'][method][scope+'_latency_ms_median']=float(np.median(timings))
    assert digest(base)==before,'Frozen base changed'
    report['frozen_base_unchanged']=True
    (OUT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
