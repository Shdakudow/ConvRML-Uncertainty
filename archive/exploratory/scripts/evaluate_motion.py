import sys,json, argparse
p=argparse.ArgumentParser(); p.add_argument("--method",choices=["endpoint","uniform"],required=True); args=p.parse_args()
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from omegaconf import OmegaConf
from dataset import get_loader
from models.convnext import QuantileConvRecon
from torchmetrics.image import StructuralSimilarityIndexMeasure
out=Path(f'results/motion_{args.method}');out.mkdir(parents=True,exist_ok=True)
ck=torch.load(f'checkpoints/motion_{args.method}_gn/best_model.pth',map_location='cpu',weights_only=False)
c=OmegaConf.create(ck['config']);c.gpu_setup.num_workers=2
_,_,loader=get_loader(c)
m=QuantileConvRecon(3,64,64,model_size=c.model.size,drop_path_rate=c.model.drop_path_rate,layer_scale_init_value=c.model.layer_scale_init_value,decoder_channels=c.model.decoder_channels,quantile_normalization=c.quantile.normalization,quantile_embedding_dim=c.quantile.embedding_dim,quantile_condition_channels=c.quantile.condition_channels).cuda()
m.load_state_dict(ck['model_state_dict']);m.eval()
original={k:v.clone() for k,v in m.state_dict().items()}
def measure(mode,limit=None):
 m.load_state_dict(original);m.eval()
 if mode=='batch_stats':
  for layer in m.quantile_head.modules():
   if isinstance(layer,torch.nn.BatchNorm2d):layer.train()
 sums=np.zeros(6);count=0;ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda()
 with torch.inference_mode():
  for i,(y,x,names) in enumerate(loader):
   if limit is not None and i>=limit:break
   y,x=y.cuda(),x.cuda();q=torch.full((len(y),),.05,device='cuda')
   r,(lo,hi)=m.forward_quantiles(y,[q,torch.full_like(q,.95)])
   vals=[((r-x)**2).mean(),(x<=lo).float().mean(),(x<=hi).float().mean(),((x>=lo)&(x<=hi)).float().mean(),(lo>hi).float().mean(),(hi-lo).mean()]
   sums+=np.array([v.item() for v in vals])*len(y);count+=len(y);ssim.update(r,x)
   if i==0 and mode=='eval':
    np.savez_compressed(out/'samples.npz',**{k:v[:10].cpu().numpy() for k,v in dict(y=y,x=x,r=r,lo=lo,hi=hi).items()},names=np.array(names[:10]))
 vals=sums/count
 return dict(zip(['mse','lower_cdf','upper_cdf','interval_coverage','crossing','mean_signed_width'],vals))|{'psnr':float(-10*np.log10(vals[0])),'ssim':ssim.compute().item(),'count':count}
report={'checkpoint_epoch':ck['epoch']+1,'test':measure('eval')}
(out/'diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
