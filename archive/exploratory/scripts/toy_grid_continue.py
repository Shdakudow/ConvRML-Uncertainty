"""Validation-selected continuation, with a fresh paired test comparison."""
import sys,json,copy,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from models.convnext import QuantileConvRecon
from models.quantile_grid import GridHead,fresh_head,pinball,LEVELS
from toy_dataset import FourierCropDataset
from scripts.toy_grid_compare import digest

OUT=Path('results/toy_grid_continue')
SOURCE=Path('results/toy_grid_large')

def main():
    if (OUT/'latest.pt').exists() or (OUT/'results.json').exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(20260922)
    ck=torch.load('checkpoints/motion_uniform_gn/best_model.pth',map_location='cpu',weights_only=False)
    c=OmegaConf.create(ck['config'])
    base=QuantileConvRecon(3,64,64,model_size=c.model.size,drop_path_rate=c.model.drop_path_rate,layer_scale_init_value=c.model.layer_scale_init_value,decoder_channels=c.model.decoder_channels,quantile_normalization='group',quantile_embedding_dim=c.quantile.embedding_dim,quantile_condition_channels=c.quantile.condition_channels)
    base.load_state_dict(ck['model_state_dict'],strict=True);base.eval().requires_grad_(False).cuda();before=digest(base)
    saved=torch.load(SOURCE/'latest.pt',map_location='cpu',weights_only=False)
    grid=GridHead(fresh_head(base.decoder.FNN1007).cpu()).cuda();grid.load_state_dict(saved['grid'])
    optimizer=torch.optim.AdamW(grid.parameters(),lr=1e-4,weight_decay=.001)
    optimizer.load_state_dict(saved['optimizers']['grid'])
    datasets={role:FourierCropDataset('/share/monakhova/shamus_data/parallel_100k',offset,count,64,48,seed,'motion') for role,offset,count,seed in [('train',32000,8192,50260921),('val',40192,1024,60260921),('test',44000,1024,80260922)]}
    prior=json.loads((SOURCE/'manifest.json').read_text())
    assert datasets['train'].files==prior['train'] and datasets['val'].files==prior['val']
    assert not set(datasets['test'].files)&set(sum(prior.values(),[]))
    (OUT/'manifest.json').write_text(json.dumps({k:d.files for k,d in datasets.items()},indent=2))
    data={}
    for role in ('train','val'):
        batches=list(DataLoader(datasets[role],batch_size=32,num_workers=2))
        data[role]=tuple(torch.cat([b[i] for b in batches]) for i in (0,1))
    def validation():
        grid.eval();total=0.
        with torch.inference_mode():
            for y,x in zip(data['val'][0].split(8),data['val'][1].split(8)):
                total+=pinball(grid(base._encode_fused(y.cuda())),x.cuda()).item()*len(y)
        return total/len(data['val'][0])
    best=validation();initial=best;best_epoch=8;stale=0;history=[]
    torch.save({'grid':grid.state_dict(),'epoch':8,'validation_pinball':best},OUT/'best.pt')
    start=time.monotonic()
    for step in range(24):
        lr=1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*step/23))
        for group in optimizer.param_groups:group['lr']=lr
        grid.train();losses=[]
        order=torch.randperm(8192,generator=torch.Generator().manual_seed(108+step))
        for idx in order.split(8):
            y,x=[v[idx].cuda() for v in data['train']]
            with torch.no_grad():f=base._encode_fused(y)
            optimizer.zero_grad(set_to_none=True);loss=pinball(grid(f),x);loss.backward()
            torch.nn.utils.clip_grad_norm_(grid.parameters(),1.);optimizer.step();losses.append(loss.item())
        val=validation();epoch=9+step
        if val<best:
            best=val;best_epoch=epoch;stale=0
            torch.save({'grid':grid.state_dict(),'epoch':epoch,'validation_pinball':best},OUT/'best.pt')
        else:stale+=1
        row={'epoch':epoch,'lr':lr,'train_pinball':float(np.mean(losses)),'validation_pinball':val,'best_epoch':best_epoch,'seconds':time.monotonic()-start}
        history.append(row);print(json.dumps(row),flush=True)
        torch.save({'grid':grid.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'history':history},OUT/'latest.tmp');(OUT/'latest.tmp').replace(OUT/'latest.pt')
        (OUT/'history.json').write_text(json.dumps(history,indent=2))
        if stale>=5:break
    report={'initial_validation':initial,'best_validation':best,'best_epoch':best_epoch,'additional_epochs':len(history),'training_and_validation_seconds':time.monotonic()-start,'history':history,'test':{},'note':'Same fresh 1024-image test for original grid, continued grid, and original conditioned head. Selection uses validation only. Single seed.'}
    # Evaluate all comparators only after validation selection is complete.
    conditional=fresh_head(base.decoder.FNN1007).cuda();conditional.load_state_dict(saved['conditioned']);conditional.eval()
    base.quantile_conditioner.load_state_dict(saved['conditioner'])
    selected=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False)
    with torch.inference_mode():
        for method in ('original_grid','continued_grid','original_conditioned'):
            if method!='original_conditioned':grid.load_state_dict(saved['grid'] if method=='original_grid' else selected['grid'])
            grid.eval();tot=np.zeros(5);count=0
            for y,x,names in DataLoader(datasets['test'],batch_size=8,num_workers=2):
                y,x=y.cuda(),x.cuda();f=base._encode_fused(y)
                p=grid(f) if method!='original_conditioned' else torch.stack([conditional(base.quantile_conditioner(f,torch.full((len(y),),q,device='cuda'))) for q in LEVELS],1)
                lo,hi=p[:,0],p[:,-1]
                values=[pinball(p,x),((lo<=x)&(x<=hi)).float().mean(),(hi-lo).clamp_min(0).mean(),(p[:,:-1]>p[:,1:]).float().mean(),((base.decoder.FNN1007(f)-x)**2).mean()]
                tot+=np.array([v.item() for v in values])*len(y)
                if count<10:
                    np.savez_compressed(OUT/f'{method}_examples_{count}.npz',y=y.cpu().numpy(),x=x.cpu().numpy(),quantiles=p.cpu().numpy())
                count+=len(y)
            report['test'][method]=dict(zip(('pinball','coverage','width','adjacent_crossing','mse'),(tot/count).tolist()))
    # The unused base quantile conditioner was loaded for comparator evaluation;
    # restore it before checking the original full frozen model digest.
    base.quantile_conditioner.load_state_dict({k.removeprefix('quantile_conditioner.'):v for k,v in ck['model_state_dict'].items() if k.startswith('quantile_conditioner.')})
    assert digest(base)==before
    report['frozen_base_unchanged']=True
    (OUT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
