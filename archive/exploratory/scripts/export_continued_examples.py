from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from skimage.metrics import structural_similarity
out=Path('/Users/shdakudow/Downloads/ConvRML_Continued_10_Examples');out.mkdir(parents=True,exist_ok=True)
def read(folder):
 arrays=[np.load(p) for p in sorted(Path(folder).glob('samples_batch_*.npz'))]
 return {k:np.concatenate([a[k] for a in arrays]) for k in arrays[0].files}
a=read('results/real_continue3h');b=read('results/real_continue3h_baseline')
assert len(a['names'])==10 and np.array_equal(a['names'],b['names']) and np.array_equal(a['x'],b['x'])
rows=[]
with PdfPages(out/'before_after_10_examples.pdf') as pdf:
 for i,name in enumerate(a['names']):
  x,r,lo,hi=[a[k][i].transpose(1,2,0) for k in ['x','r','lo','hi']];old=b['r'][i].transpose(1,2,0)
  def metrics(p):return (-10*np.log10(np.mean((p-x)**2)),structural_similarity(x,p,data_range=1.,channel_axis=-1))
  p0,s0=metrics(old);p1,s1=metrics(r);cov=((x>=lo)&(x<=hi)).mean()
  rows.append(dict(sample=i+1,id=str(name),before_psnr=float(p0),after_psnr=float(p1),before_ssim=float(s0),after_ssim=float(s1),coverage=float(cov)))
  err=np.abs(r-x).mean(-1);width=(hi-lo).mean(-1);cross=(lo>hi).mean(-1)
  fig,axs=plt.subplots(2,4,figsize=(14,8));fig.suptitle(f'{name} | PSNR {p0:.2f} → {p1:.2f} dB | SSIM {s0:.3f} → {s1:.3f} | coverage {cov:.1%}',fontsize=13)
  for ax,img,title in zip(axs.flat,[x,old,r,err,lo,hi,width,cross],['Ground truth','Before: epoch 12','After: epoch 30','After: absolute error','Lower q=0.05','Upper q=0.95','Signed interval width','Crossed RGB fraction']):
   if img.ndim==3:ax.imshow(np.clip(img,0,1))
   else:
    if title.startswith('Signed'):
     lim=max(float(np.abs(img).max()),1e-6);kw=dict(cmap='coolwarm',vmin=-lim,vmax=lim)
    elif title.startswith('Crossed'):kw=dict(cmap='Reds',vmin=0,vmax=1)
    else:kw=dict(cmap='magma',vmin=0,vmax=max(float(img.max()),1e-6))
    im=ax.imshow(img,**kw);fig.colorbar(im,ax=ax,fraction=.045,pad=.03)
   ax.set_title(title,fontsize=11);ax.axis('off')
  fig.text(.03,.025,'Supervised 239×239 region only. First 10 retained test IDs; IDs 5/6 excluded, difficult ID7 retained. RGB clipped for display only; intervals uncalibrated.',fontsize=9)
  fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(out/f'comparison_{i+1:02d}.png',dpi=160);pdf.savefig(fig);plt.close(fig)
  for label,img in [('reconstruction',r),('ground_truth',x),('before',old)]:Image.fromarray((np.clip(img,0,1)*255).round().astype('uint8')).save(out/f'{label}_{i+1:02d}.png')
(out/'sample_metrics.json').write_text(json.dumps(rows,indent=2))
(out/'README.txt').write_text('Ten first retained test images: IDs 0,1,2,3,4,7,8,9,10,11. Selected by fixed ordering, not favorable scores. The difficult former example8 (ID7) is now comparison06. Reconstructions and targets are cropped to the supervised 239x239 region. The original full measurement arrays remain in /Users/shdakudow/ConvRML/results/real_continue3h. PDF and PNGs compare epoch12 with epoch30. No calibrated guarantee.\n')
print(out)
