"""Compact RGB image toy, 50 total epochs, native uncertainty mechanisms.

One GPU per two-epoch job; eight compact networks trained sequentially.
Fixed model counts 25 backbone + 25 head epochs, rather than 50+50.
"""
import argparse,copy,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.small7k_train import gaussian_crps,sample_crps
from scripts.small7k_eval import gaussian_mixture_crps,mixture_quantiles
from baselines.mc_dropout import dropout_inference

ROOT=Path('results/toy50_hour_20260926')
NAMES=[f'ensemble_{i}' for i in range(5)]+['mc','q','fixed']

class Block(nn.Module):
    def __init__(self,cin,cout,drop=False):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(4,cout),nn.GELU(),
             nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(4,cout),nn.GELU(),nn.Dropout2d(.1) if drop else nn.Identity())
    def forward(self,x):return self.layers(x)

class Backbone(nn.Module):
    def __init__(self,drop=False):
        super().__init__();self.e1=Block(3,16);self.e2=Block(16,32);self.middle=Block(32,64,drop)
        self.d2=Block(96,32,drop);self.d1=Block(48,16,drop);self.head=nn.Conv2d(16,3,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def features(self,y):
        a=self.e1(y);b=self.e2(F.avg_pool2d(a,2));c=self.middle(F.avg_pool2d(b,2))
        d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
    def forward(self,y):return y+self.head(self.features(y))

class Model(nn.Module):
    def __init__(self,name):
        super().__init__();self.name=name;self.base=Backbone(name=='mc');self.head_stage=False
        if name.startswith('ensemble'):
            self.variance=nn.Conv2d(16,3,3,padding=1);nn.init.zeros_(self.variance.weight);nn.init.constant_(self.variance.bias,-4.6)
        if name=='q':
            self.embedding=nn.Sequential(nn.Linear(1,16),nn.SiLU(),nn.Linear(16,32));nn.init.zeros_(self.embedding[-1].weight);nn.init.zeros_(self.embedding[-1].bias)
            self.quantile_head=copy.deepcopy(self.base.head)
        if name=='fixed':self.quantile_head=nn.Conv2d(16,6,3,padding=1)
    def train(self,mode=True):
        super().train(mode)
        if self.head_stage:self.base.eval()
        return self
    def freeze(self,initialize):
        if initialize:
            with torch.no_grad():
                self.quantile_head.weight.copy_(self.base.head.weight.repeat(2,1,1,1));self.quantile_head.bias.copy_(self.base.head.bias.repeat(2))
        self.base.requires_grad_(False);self.head_stage=True;self.base.eval()
    def features(self,y):
        if self.head_stage:
            with torch.no_grad():f=self.base.features(y);r=y+self.base.head(f)
        else:f=self.base.features(y);r=y+self.base.head(f)
        return f,r
    def quantile(self,y,f,q):
        scale,shift=self.embedding(q[:,None]).chunk(2,1)
        return y+self.quantile_head(f*(1+scale[:,:,None,None])+shift[:,:,None,None])
    def forward(self,y):return self.base(y)

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'data.pt').exists():return
    src=Path('data/formal_rml_25k');cal=Path('data/formal_rml_calibration95')
    manifest=json.loads((src/'manifest.json').read_text());cm=json.loads((cal/'manifest.json').read_text())
    rng=np.random.default_rng(97126)
    selected={'train':rng.permutation(20000)[:512].tolist(),'val':(20000+rng.permutation(2500)[:128]).tolist(),
              'test':list(range(24000,24256)),'calibration':list(range(256))}
    records=manifest['train']+manifest['val']+manifest['test'];seen=[];data={}
    for role,indices in selected.items():
        pairs=np.load((cal if role=='calibration' else src)/'pairs.npy',mmap_mode='r')
        images=[]
        for s in range(0,len(indices),32):
            v=torch.from_numpy(np.array(pairs[indices[s:s+32]],copy=True)).float()/255
            images.append(F.interpolate(v.flatten(0,1),size=(64,64),mode='area').reshape(-1,2,3,64,64))
        data[role]=torch.cat(images)
        seen.extend([(cm['calibration'][i] if role=='calibration' else records[i]) for i in indices])
    assert len({v['id'] for v in seen})==len(seen) and len({v['target_sha256'] for v in seen})==len(seen)
    protocol={'split':{k:len(v) for k,v in selected.items()},'source_indices':selected,'resolution':64,'epochs':50,
              'fixed_stages':'25 reconstruction epochs + 25 frozen-backbone endpoint-head epochs = 50 total',
              'ensemble_members':5,'mc_train_draws':4,'mc_inference_draws':30,'quantile_training':'MSE + 0.5 * estimated quantile-integral CRPS = MSE + uniform-q pinball; quantile crossings checked at evaluation',
              'architecture':'Compact 16/32/64-channel U-Net, shared architecture across methods, decoder dropout .1 for MC; not full ConvRML',
              'training':'From scratch, exact final epoch50, AdamW lr0.001 wd0.0001, batch32, FP32, validation each two-epoch boundary, no early stopping or test tuning',
              'calibration':'256 disjoint calibration images; pooled pixel scale about interval midpoint for95% target; empirical marginal coverage only',
              'limits':'One seed per method, five independent ensemble members. Same data/50 total epochs, unequal compute. Reused real-RML images downsampled to64; conclusions are toy-scale only. GPU time target <1 hour excludes queue delays.'}
    torch.save(data,ROOT/'data.pt');(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2));print(protocol,flush=True)

def pinball(pred,x,q):
    e=x-pred;q=q[:,None,None,None];return torch.maximum(q*e,(q-1)*e).mean()

def objective(model,y,x,validation=False):
    name=model.name
    if name=='mc':
        with dropout_inference(model) if validation else torch.enable_grad():
            draws=torch.stack([model(y) for _ in range(4)])
        return sample_crps(draws,x,unbiased=True).mean()
    f,r=model.features(y)
    if name.startswith('ensemble'):return gaussian_crps(r,F.softplus(model.variance(f))+1e-6,x).mean()
    if name=='fixed':
        if not model.head_stage:return (r-x).square().mean()
        p=model.quantile_head(f).reshape(len(y),2,3,64,64)+y[:,None]
        return sum(pinball(p[:,j],x,y.new_full((len(y),),q)) for j,q in enumerate([.025,.975]))/2
    if validation:
        loss=sum(pinball(model.quantile(y,f,y.new_full((len(y),),q)),x,y.new_full((len(y),),q)) for q in [.1,.3,.5,.7,.9])/5
    else:
        q=torch.rand(len(y),device=y.device);loss=pinball(model.quantile(y,f,q),x,q)
    return (r-x).square().mean()+loss

def make(name):
    torch.manual_seed(12697+(int(name[-1]) if name.startswith('ensemble') else 0))
    return Model(name).cuda()

def smoke():
    data=torch.load(ROOT/'data.pt',weights_only=True)['train'][:32].cuda();timings={}
    for name in ['ensemble_0','mc','q','fixed','fixed_head']:
        model=make('fixed' if name=='fixed_head' else name)
        if name=='fixed_head':model.freeze(True)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        for step in range(12):
            if step==3:torch.cuda.synchronize();start=time.monotonic()
            opt.zero_grad(set_to_none=True);loss=objective(model,data[:,0],data[:,1]);assert torch.isfinite(loss);loss.backward();opt.step()
        torch.cuda.synchronize();timings[name]=(time.monotonic()-start)/9
        del model,opt
    steps=512/32
    estimate=steps*(50*(5*timings['ensemble_0']+timings['mc']+timings['q'])+25*(timings['fixed']+timings['fixed_head']))
    report={'seconds_per_batch':timings,'estimated_training_seconds':estimate,'estimated_with_validation_io_evaluation_seconds':estimate*1.6+180,'gpu':torch.cuda.get_device_name(),'passed':estimate*1.6+180<3600}
    (ROOT/'smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    assert report['passed'],'Runtime gate exceeded; reduce toy size before submitting training'

def train(end_epoch):
    assert json.loads((ROOT/'smoke.json').read_text())['passed']
    data={k:v.cuda() for k,v in torch.load(ROOT/'data.pt',weights_only=True).items() if k in ['train','val']}
    for name in NAMES:
        out=ROOT/name;out.mkdir(exist_ok=True);model=make(name);history=[];start_epoch=0
        ck=torch.load(out/'latest.pt',weights_only=False,map_location='cpu') if (out/'latest.pt').exists() else None
        if ck:
            start_epoch=ck['epoch'];history=ck['history'];model.load_state_dict(ck['model'])
            if name=='fixed' and start_epoch>=26:model.freeze(False)
        assert 0<=end_epoch-start_epoch<=2
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
        if ck:
            opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state(ck['cuda_rng']);del ck
        for epoch in range(start_epoch+1,end_epoch+1):
            if name=='fixed' and epoch==26:
                model.freeze(True);opt=torch.optim.AdamW(model.quantile_head.parameters(),lr=.001,weight_decay=.0001)
            order=torch.randperm(512,generator=torch.Generator().manual_seed(876+epoch)).cuda();model.train();total=0.
            torch.cuda.synchronize();tick=time.monotonic()
            for ix in order.split(32):
                b=data['train'][ix];opt.zero_grad(set_to_none=True);loss=objective(model,b[:,0],b[:,1]);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(b)
            model.eval();val=None
            if epoch%2==0 or epoch==25:
                with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.cuda.manual_seed(762);val=sum(objective(model,b[:,0],b[:,1],True).item()*len(b) for b in data['val'].split(32))/128
            torch.cuda.synchronize();row={'epoch':epoch,'training_loss':total/512,'validation_loss':val,'seconds':time.monotonic()-tick,'stage':'head' if model.head_stage else 'backbone'};history.append(row)
            torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},out/'latest.tmp');(out/'latest.tmp').replace(out/'latest.pt')
            (out/'history.json').write_text(json.dumps(history,indent=2));print(name,json.dumps(row),flush=True)
        del model,opt

@torch.inference_mode()
def predict(name,role):
    data=torch.load(ROOT/'data.pt',weights_only=True)[role];models=[]
    for member in [f'ensemble_{i}' for i in range(5)] if name=='ensemble' else [name]:
        model=make(member);ck=torch.load(ROOT/member/'latest.pt',map_location='cpu',weights_only=False);assert ck['epoch']==50
        model.load_state_dict(ck['model']);model.eval();models.append(model)
    means=[];params=[];qlevels=(torch.arange(31,device='cuda')+.5)/31
    torch.manual_seed(1527 if role=='calibration' else 1752)
    for batch in data.split(16):
        y=batch[:,0].cuda()
        if name=='ensemble':
            p=[];m=[]
            for model in models:
                f,r=model.features(y);m.append(r);p.append(torch.stack([r,F.softplus(model.variance(f))+1e-6],1))
            r=torch.stack(m).mean(0);v=torch.stack(p,1)
        elif name=='mc':
            with dropout_inference(models[0]):draws=torch.stack([models[0](y) for _ in range(30)])
            r=draws.mean(0);v=draws.transpose(0,1)
        else:
            model=models[0];f,r=model.features(y)
            if name=='fixed':v=model.quantile_head(f).reshape(len(y),2,3,64,64)+y[:,None]
            else:v=torch.stack([model.quantile(y,f,y.new_full((len(y),),float(q))) for q in qlevels],1)
        means.append(r.cpu());params.append(v.cpu())
    return torch.cat(means),torch.cat(params)

def intervals_score(name,p,x):
    if name=='ensemble':
        mu=p[:,:,0].transpose(0,1).flatten(2);var=p[:,:,1].transpose(0,1).flatten(2)
        q=mixture_quantiles(mu,var,[.025,.975]);lo,hi=q[:,0],q[:,1]
        score=gaussian_mixture_crps(mu,var,x.flatten(1));cross=None
    elif name=='mc':
        draws=p.transpose(0,1).flatten(2);q=torch.quantile(draws,torch.tensor([.025,.975]),dim=0);lo,hi=q[0],q[1]
        score=sample_crps(draws,x.flatten(1));cross=None
    elif name=='q':
        values=p.flatten(2);cross=(values[:,1:]<values[:,:-1]).float().mean().item()
        ordered=values.sort(1).values;q=torch.quantile(ordered,torch.tensor([.025,.975]),dim=1);lo,hi=q[0],q[1]
        score=sample_crps(ordered.transpose(0,1),x.flatten(1))
    else:
        values=p.flatten(2);lo=values.min(1).values;hi=values.max(1).values;score=None;cross=(values[:,0]>values[:,1]).float().mean().item()
    return lo,hi,score,cross

def evaluate():
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    data=torch.load(ROOT/'data.pt',weights_only=True);results={};examples={};factors={}
    # GPU inference first; evaluation uses small CPU chunks.
    for name in ['ensemble','mc','q','fixed']:
        tick=time.monotonic();mean,params=predict(name,'calibration');scores=[]
        for s in range(0,256,8):
            x=data['calibration'][s:s+8,1];lo,hi,_,_=intervals_score(name,params[s:s+8],x);mid=(lo+hi)/2;rad=((hi-lo)/2).clamp_min(1e-6)
            scores.append(((x.flatten(1)-mid).abs()/rad).numpy())
        factor=float(np.quantile(np.concatenate(scores),.95,method='higher'));factors[name]=factor
        mean,params=predict(name,'test');x=data['test'][:,1]
        rows=[];crosses=[]
        for s in range(0,256,8):
            target=x[s:s+8];r=mean[s:s+8];lo,hi,score,cross=intervals_score(name,params[s:s+8],target);mid=(lo+hi)/2;rad=((hi-lo)/2).clamp_min(1e-6)
            cl,ch=mid-factor*rad,mid+factor*rad;t=target.flatten(1)
            psnr=-10*torch.log10((r.double()-target.double()).square().mean((1,2,3)));sim=ssim(r,target,data_range=1.,reduction='none')
            for i in range(len(r)):
                rows.append({'psnr':psnr[i].item(),'ssim':sim[i].item(),'crps':score[i].mean().item() if score is not None else None,
                             'raw_coverage':((lo[i]<=t[i])&(t[i]<=hi[i])).float().mean().item(),'raw_width':(hi[i]-lo[i]).mean().item(),
                             'calibrated_coverage':((cl[i]<=t[i])&(t[i]<=ch[i])).float().mean().item(),'calibrated_width':(ch[i]-cl[i]).mean().item(),
                             'interval_score95':((ch[i]-cl[i])+40*(cl[i]-t[i]).clamp_min(0)+40*(t[i]-ch[i]).clamp_min(0)).mean().item()})
            if cross is not None:crosses.append(cross)
            if s==0:examples.update({name+'_mean':r[:6].numpy(),name+'_width':(ch-cl).reshape(-1,3,64,64)[:6].mean(1).numpy()})
        results[name]={key:float(np.mean([r[key] for r in rows])) if rows[0][key] is not None else None for key in rows[0]}
        results[name].update(calibration_factor=factor,crossing_rate=float(np.mean(crosses)) if crosses else None,evaluation_seconds=time.monotonic()-tick)
        print(name,json.dumps(results[name]),flush=True)
    training={name:sum(v['seconds'] for v in json.loads((ROOT/name/'history.json').read_text())) for name in NAMES}
    (ROOT/'comparison.json').write_text(json.dumps({'results':results,'training_seconds':training,'protocol':json.loads((ROOT/'protocol.json').read_text()),'notes':['Q CRPS approximated by sorted31 midpoint quantiles; no tail extrapolation. Fixed endpoints: interval score only. Raw CRPS before interval calibration. All-pixel evaluation; no independent-pixel coverage guarantee.']},indent=2))
    lines=['# Compact 50-epoch toy comparison','','| Method | PSNR | SSIM | Raw CRPS | Calibrated coverage95 | Calibrated width |','|---|---:|---:|---:|---:|---:|']
    for name,m in results.items():
        crps=f"{m['crps']:.6f}" if m['crps'] is not None else 'N/A'
        lines.append(f"| {name} | {m['psnr']:.3f} | {m['ssim']:.4f} | {crps} | {m['calibrated_coverage']:.2%} | {m['calibrated_width']:.5f} |")
    lines+=['','Fixed head: 25 backbone + 25 head epochs. Other networks: 50 epochs each; ensemble has five networks. Compact64px toy, not full ConvRML.','Total measured training/validation seconds: '+str(sum(training.values()))]
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
    examples.update(measurement=data['test'][:6,0].numpy(),target=data['test'][:6,1].numpy());np.savez_compressed(ROOT/'examples.npz',**examples)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    vmax=float(np.quantile(np.stack([examples[n+'_width'] for n in results]),.995))
    with PdfPages(ROOT/'comparison_examples.pdf') as pdf:
        for i in range(6):
            fig,axes=plt.subplots(4,4,figsize=(11,10))
            for row,name in enumerate(results):
                for col,v in enumerate([examples['measurement'][i],examples['target'][i],examples[name+'_mean'][i],examples[name+'_width'][i]]):
                    ax=axes[row,col];ax.axis('off')
                    if col<3:ax.imshow(v.transpose(1,2,0).clip(0,1))
                    else:im=ax.imshow(v,cmap='magma',vmin=0,vmax=max(vmax,1e-6))
                    if row==0:ax.set_title(['Measurement','Ground truth','Prediction','95% interval width'][col])
                axes[row,0].text(-.06,.5,name,transform=axes[row,0].transAxes,ha='right')
            fig.suptitle(f'Compact 50-epoch toy | example {i+1} | independently calibrated95%')
            fig.subplots_adjust(left=.12,right=.92,top=.93,bottom=.04);fig.colorbar(im,ax=axes[:,-1],fraction=.025,pad=.02)
            pdf.savefig(fig);fig.savefig(ROOT/f'example_{i+1}.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train','evaluate']);p.add_argument('--epochs',type=int);a=p.parse_args()
    torch.set_num_threads(2)
    if a.action=='prepare':prepare()
    elif a.action=='smoke':smoke()
    elif a.action=='train':train(a.epochs)
    else:evaluate()
