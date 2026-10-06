"""Isolated from-scratch 5k/1k/1k native-method comparison."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.shamus_rml import ShamusRML
from models.formal_quantile import QuantileRML, pinball, LEVELS
from baselines.formal_rml import FormalRML
from baselines.deep_ensemble import DeepEnsembleMember

ROOT=Path('results/native_uncertainty_5k_20260926')
DATA=Path('data/native_uncertainty_5k_20260926')
SOURCE=Path('data/formal_rml_25k')
KINDS={**{i:'ensemble' for i in range(5)},5:'mc',6:'normal',7:'conditioned',8:'fixed',
       **{i:'ensemble_crps' for i in range(9,14)},14:'mc_crps'}

def gaussian_crps(mean,variance,target):
    sigma=variance.clamp_min(1e-8).sqrt();z=(target-mean)/sigma
    phi=torch.exp(-.5*z.square())/(2*torch.pi)**.5
    return sigma*(z*torch.erf(z/2**.5)+2*phi-1/torch.pi**.5)

def sample_crps(samples,target,unbiased=False):
    # Sorted-sample identity avoids a T-by-T-by-image tensor.
    n=len(samples);ordered=samples.sort(dim=0).values
    coefficient=(2*torch.arange(1,n+1,device=samples.device,dtype=samples.dtype)-n-1).reshape((n,)+(1,)*(samples.ndim-1))
    denominator=n*(n-1) if unbiased else n*n
    if denominator==0:raise ValueError('At least two draws required for unbiased CRPS')
    return (samples-target).abs().mean(0)-(coefficient*ordered).sum(0)/denominator

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    DATA.mkdir(parents=True,exist_ok=True);ROOT.mkdir(parents=True,exist_ok=True)
    if (DATA/'manifest.json').exists():
        manifest=json.loads((DATA/'manifest.json').read_text())
        assert manifest['source_manifest_sha256']==sha(SOURCE/'manifest.json')
        return
    original=json.loads((SOURCE/'manifest.json').read_text())
    rng=np.random.default_rng(20260926)
    selection={'train':rng.permutation(20000)[:5000].tolist(),
               'val':(20000+rng.permutation(2500)[:1000]).tolist(),
               'test':list(range(24000,25000))}
    rows=original['train']+original['val']+original['test']
    chosen=[rows[i] for values in selection.values() for i in values]
    assert len({r['id'] for r in chosen})==7000
    assert len({r['target_sha256'] for r in chosen})==7000
    pairs=np.load(SOURCE/'pairs.npy',mmap_mode='r')
    assert pairs.shape==(25000,2,3,256,256)
    for index in [values[j] for values in selection.values() for j in [0,1,2,-1]]:
        assert hashlib.sha256(pairs[index,1].tobytes()).hexdigest()==rows[index]['target_sha256']
    manifest={'seed':20260926,'source':str(SOURCE),'source_manifest_sha256':sha(SOURCE/'manifest.json'),
              'indices':selection,**{k:[rows[i] for i in values] for k,values in selection.items()},
              'notes':['5k train / 1k validation / 1k test; preserve existing split boundaries.',
                       'Test is last 1000 of original 2500, not the first1000 repeatedly compared recently. It is not a certified unseen dataset across all earlier experiments.',
                       'Exact processed target and ID overlap excluded; near-duplicate acquisition groups not certified.',
                       'Same cached RGB256 preprocessing. No pretrained or prior experiment weights.',
                       'Native training losses and raw uncertainty; no calibration fitting. Fixed head has a separate frozen-backbone stage.']}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Prepared 5000/1000/1000 disjoint records',flush=True)

class Pairs(Dataset):
    def __init__(self,role):
        manifest=json.loads((DATA/'manifest.json').read_text())
        self.indices=manifest['indices'][role]
        self.a=np.load(Path(manifest['source'])/'pairs.npy',mmap_mode='r')
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return torch.from_numpy(np.array(self.a[self.indices[i]],copy=True)).float()/255

def model_for(worker):
    kind=KINDS[worker]
    if kind.startswith('ensemble'):return DeepEnsembleMember(FormalRML('ensemble'))
    if kind.startswith('mc'):return FormalRML('mc')
    base=ShamusRML()
    if kind=='normal':return base
    if kind=='fixed':
        ck=torch.load(ROOT/'worker_6/best.pt',map_location='cpu',weights_only=False)
        assert ck['manifest_sha256']==sha(DATA/'manifest.json')
        base.load_state_dict(ck['model'])
    return QuantileRML(base,kind)

def digest(model):
    h=hashlib.sha256()
    for value in model.state_dict().values():h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def train(worker,end_epoch,smoke=False):
    torch.set_num_threads(2);seed=20260926+(worker if worker<5 else worker-9 if 9<=worker<=13 else 0)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    kind=KINDS[worker];out=ROOT/f'worker_{worker}';out.mkdir(parents=True,exist_ok=True)
    model=model_for(worker).cuda();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    trainset=Pairs('train');valset=Pairs('val');history=[];best=float('inf');start_epoch=0
    qgen=torch.Generator(device='cuda').manual_seed(1837)
    if (out/'latest.pt').exists() and not smoke:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['manifest_sha256']==sha(DATA/'manifest.json') and ck['kind']==kind
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];best=ck['best'];start_epoch=ck['epoch']
        torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state(ck['cuda_rng']);qgen.set_state(ck['q_rng']);del ck
    assert smoke or 0<=end_epoch-start_epoch<=2,'At most two epochs per job'
    frozen=digest(model.base) if kind=='fixed' else None
    def loss(batch,validation=False):
        y,x=batch[:,0].cuda(),batch[:,1].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if kind.startswith('ensemble'):
                mean,var=model(y);mean,var=mean.float(),var.float()
                if kind=='ensemble_crps':return gaussian_crps(mean,var,x).mean()
                return .5*(var.log()+(mean-x).square()/var).mean()
            if kind=='mc_crps':
                from baselines.mc_dropout import dropout_inference
                # Four independent dropout draws, U-statistic estimate of CRPS.
                # Keep dropout active during validation without enabling other training behavior.
                if validation:
                    with dropout_inference(model):samples=torch.stack([model(y).float() for _ in range(4)])
                else:samples=torch.stack([model(y).float() for _ in range(4)])
                return sample_crps(samples,x,unbiased=True).mean()
            if kind in ['normal','mc']:return (model(y).float()-x).square().mean()
            f,r=model.features(y)
            if kind=='fixed':
                p=model.predict(y,f)
                return sum(pinball(p[:,j].float(),x,x.new_full((len(y),),q)) for j,q in enumerate(LEVELS))/2
            if validation:
                return sum(pinball(model.predict(y,f,x.new_full((len(y),),q)).float(),x,x.new_full((len(y),),q)) for q in LEVELS)/2
            q=torch.rand(len(y),device='cuda',generator=qgen)
            return pinball(model.predict(y,f,q).float(),x,q)+(r.float()-x).square().mean()
    print(json.dumps({'kind':kind,'start_epoch':start_epoch,'end_epoch':end_epoch,'smoke':smoke,'seed':seed,'train_count':len(trainset)}),flush=True)
    if smoke:
        model.train();loader=DataLoader(trainset,batch_size=2,num_workers=2)
        tick=None
        for i,batch in enumerate(loader):
            opt.zero_grad(set_to_none=True);v=loss(batch);assert torch.isfinite(v);v.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if i==4:torch.cuda.synchronize();tick=time.monotonic()
            if i==14:break
        torch.cuda.synchronize();seconds=(time.monotonic()-tick)/10
        report={'passed':True,'kind':kind,'seconds_per_batch':seconds,'estimated_training_seconds_per_epoch':seconds*len(trainset)/2,'gpu':torch.cuda.get_device_name()}
        (ROOT/f'smoke_{kind}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
    if kind!='fixed':
        smoke_report=json.loads((ROOT/f'smoke_{kind}.json').read_text())
        assert smoke_report['passed'] and smoke_report['estimated_training_seconds_per_epoch']*2*1.5<3*3600
    for epoch in range(start_epoch+1,end_epoch+1):
        loader=DataLoader(trainset,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(8173+epoch),num_workers=2,pin_memory=True)
        tick=time.monotonic();model.train();total=0.
        for batch in loader:
            opt.zero_grad(set_to_none=True);v=loss(batch);assert torch.isfinite(v);v.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=v.item()*len(batch)
        model.eval();validation=0.
        with torch.inference_mode():
            for index,batch in enumerate(DataLoader(valset,batch_size=2,num_workers=2,pin_memory=True)):
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.cuda.manual_seed(99173+index)
                    v=loss(batch,True)
                assert torch.isfinite(v);validation+=v.item()*len(batch)
        validation/=len(valset)
        if frozen:assert digest(model.base)==frozen
        row={'epoch':epoch,'train_loss':total/len(trainset),'validation_loss':validation,'seconds':time.monotonic()-tick};history.append(row)
        metadata={'kind':kind,'epoch':epoch,'manifest_sha256':sha(DATA/'manifest.json')}
        if validation<best:
            best=validation;torch.save({**metadata,'model':model.state_dict(),'validation':best},out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
        torch.save({**metadata,'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'best':best,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'q_rng':qgen.get_state()},out/'latest.tmp');(out/'latest.tmp').replace(out/'latest.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','smoke']);p.add_argument('--worker',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID',0)));p.add_argument('--epochs',type=int,default=2);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:train(a.worker,a.epochs,a.action=='smoke')
