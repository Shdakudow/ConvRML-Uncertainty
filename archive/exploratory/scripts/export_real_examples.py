from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from skimage.metrics import structural_similarity
src=Path('results/real_small');dst=Path('/Users/shdakudow/Downloads/ConvRML_Real_RML_Examples');dst.mkdir(parents=True,exist_ok=True)
n=0
with PdfPages(dst/'real_rml_10_examples.pdf') as pdf:
 for file in sorted(src.glob('samples_batch_*.npz')):
  a=np.load(file)
  for i,name in enumerate(a['names']):
   n+=1
   y,x,r,lo,hi=[a[k][i].transpose(1,2,0) for k in ['y','x','r','lo','hi']]
   roi=(slice(31,270),slice(128,367))
   xx,rr,ll,hh=[v[roi] for v in [x,r,lo,hi]]
   psnr=-10*np.log10(np.mean((rr-xx)**2));ssim=structural_similarity(xx,rr,data_range=1.,channel_axis=-1)
   coverage=((xx>=ll)&(xx<=hh)).mean()
   err=np.abs(r-x).mean(-1);width=(hi-lo).mean(-1);cross=(lo>hi).mean(-1)
   fig,axs=plt.subplots(2,4,figsize=(16,8));fig.suptitle(f'Test {n:02d} | {name} | ROI PSNR {psnr:.2f} dB | SSIM {ssim:.3f} | coverage {coverage:.1%}',fontsize=14)
   for ax,img,title in zip(axs.flat,[y,x,r,lo,hi,err,width,cross],['Real RML measurement','Ground truth','Reconstruction','Lower q=0.05','Upper q=0.95','Mean absolute RGB error','Signed interval width','Crossed-channel fraction']):
    if img.ndim==3:ax.imshow(np.clip(img,0,1))
    else:
     if title.startswith('Signed'):
      lim=max(float(np.abs(img).max()),1e-6);kw=dict(cmap='coolwarm',vmin=-lim,vmax=lim)
     elif title.startswith('Crossed'):kw=dict(cmap='Reds',vmin=0,vmax=1)
     else:kw=dict(cmap='magma',vmin=0,vmax=max(float(img.max()),1e-6))
     im=ax.imshow(img,**kw);fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    ax.set_title(title,fontsize=11);ax.axis('off')
   fig.text(.04,.035,'Full 300x480 images displayed; metrics use the RML ROI (rows 31:270, columns 128:367). RGB display clipped only; bounds are uncalibrated.',fontsize=10)
   fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(dst/f'comparison_{n:02d}.png',dpi=150);pdf.savefig(fig);plt.close(fig)
   for label,img in [('reconstruction',r),('ground_truth',x),('measurement',y)]:
    Image.fromarray((np.clip(img,0,1)*255).round().astype('uint8')).save(dst/f'{label}_{n:02d}.png')
assert n==10,n
print(f'Exported {n} examples to {dst}')
