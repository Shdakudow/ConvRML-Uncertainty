from pathlib import Path
import json,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import structural_similarity
root=Path('results/toy_uniform');d=json.loads((root/'diagnosis.json').read_text());a=np.load(root/'samples.npz');t=d['test'];b=d['batch_stats_first128'];e=d['eval_first128']
def page_text(pdf,title,paragraphs):
 fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.91,title,fontsize=22,weight='bold');y=.82
 for p in paragraphs:
  wrapped=textwrap.fill(p,105);fig.text(.07,y,wrapped,fontsize=11,va='top',linespacing=1.5);y-=.031*(wrapped.count('\n')+1)+.027
 pdf.savefig(fig);plt.close(fig)
with PdfPages(root/'uniform_quantile_comparison.pdf') as pdf:
 baseline=json.loads(Path('results/toy_diagnosis/diagnosis.json').read_text())['test']
 seconds=int((root/'training_seconds.txt').read_text())
 page_text(pdf,'Uniform quantile training: comparison',[
 'New approach: one u sampled independently from Uniform(0,1) per image on every training step. Inference uses u=0.05 and u=0.95. MSE reconstruction loss plus sampled pinball loss; no crossing penalty.',
 'Same ConvNeXt model and BatchNorm architecture, 8,192 training images, batch size 32, eight epochs, and RTX 6000 Ada constraint as the endpoint experiment. Separate random initializations; one run per approach. This is a practical comparison, not an isolated causal test of sampling.',
 'Uniform training plus per-epoch validation/checkpointing took %d min %d sec. Endpoint job elapsed time was 14 min 34 sec (includes job startup). Queue wait excluded; timing scopes differ slightly.'%(seconds//60,seconds%60),
 'Uniform checkpoint selected by lowest validation MSE: epoch %d. Evaluation: same 1,024 held-out images, no test-based checkpoint selection. Ten examples are the same first ten test images used in the earlier PDF.'%d['checkpoint_epoch'],
 'Endpoint baseline: PSNR %.2f dB; SSIM %.3f; coverage %.1f%%; crossings %.1f%%; mean signed width %.4f.'%(baseline['psnr'],baseline['ssim'],100*baseline['interval_coverage'],100*baseline['crossing'],baseline['mean_signed_width']),
 'Uniform approach: PSNR %.2f dB; SSIM %.3f; coverage %.1f%%; crossings %.1f%%; mean signed width %.4f.'%(t['psnr'],t['ssim'],100*t['interval_coverage'],100*t['crossing'],t['mean_signed_width']),
 'Uniform endpoint CDFs: %.1f%% / %.1f%%, targeting 5%% / 95%%. Coverage target is 90%%. Signed widths can cancel when bounds cross; they do not by themselves measure useful uncertainty.'%(100*t['lower_cdf'],100*t['upper_cdf'])
 ])
 page_text(pdf,'Interpretation and diagnostic controls',[
 'Normal inference uses stored BatchNorm statistics. Sampling different u values within each training batch changes the normalization context relative to endpoint-only training. The architecture has not been repaired or calibrated.',
 'Same first 128 test images, normal inference: coverage %.1f%%; crossings %.1f%%; endpoint CDFs %.1f%% / %.1f%%.'%(100*e['interval_coverage'],100*e['crossing'],100*e['lower_cdf'],100*e['upper_cdf']),
 'Diagnostic only, quantile-head BatchNorm using current batch statistics: coverage %.1f%%; crossings %.1f%%; endpoint CDFs %.1f%% / %.1f%%. No parameter updates. This is not a deployment recommendation.'%(100*b['interval_coverage'],100*b['crossing'],100*b['lower_cdf'],100*b['upper_cdf']),
 'Data are 64 x 64 RML-aligned ground-truth images, convolved with random Gaussian PSFs, centrally cropped to 48 x 48, and corrupted with noise. These are synthetic measurements, not captured DiffuserCam measurements.',
 'Whole-image PSNR/SSIM include large dark backgrounds. Inspect object detail and error maps as well. RGB pictures are clipped only for display; numerical metrics use raw predictions. Crossing and coverage are computed per RGB entry.',
 'Neither experiment includes held-out risk calibration. Approximate empirical 90% coverage does not establish the guarantee in Angelopoulos et al., arXiv:2202.05265. This reused test set is now an exploratory benchmark; final confirmation should use fresh held-out data.',
 'Sample pages show measurement, target, reconstruction, lower and upper bounds, error, signed interval width, and crossing fraction. Negative blue width indicates reversed bounds. Heatmap scales vary by image.'
 ])
 for i,name in enumerate(a['names']):
  x,r,y,lo,hi=[a[k][i].transpose(1,2,0) for k in ('x','r','y','lo','hi')]
  err=np.abs(r-x).mean(-1);width=(hi-lo).mean(-1);cross=(lo>hi).mean(-1);covered=((x>=lo)&(x<=hi)).mean()
  psnr=-10*np.log10(np.mean((r-x)**2));ssim=structural_similarity(x,r,data_range=1.,channel_axis=-1)
  fig,axs=plt.subplots(2,4,figsize=(14,8));fig.suptitle(f'Test {i+1:02d} | {name} | PSNR {psnr:.2f} dB | SSIM {ssim:.3f} | coverage {covered:.1%}',fontsize=15)
  for ax,img,title in zip(axs.flat,[y,x,r,lo,hi,err,width,cross],['Synthetic measurement','Ground truth','MSE reconstruction','Lower q = 0.05','Upper q = 0.95','Mean absolute RGB error','Signed width: upper - lower','Fraction of RGB channels crossed']):
   if img.ndim==3:ax.imshow(np.clip(img,0,1))
   else:
    lim=max(float(np.abs(width).max()),1e-6)
    kwargs=dict(cmap='magma',vmin=0,vmax=max(float(err.max()),1e-6)) if title.startswith('Mean') else (dict(cmap='coolwarm',vmin=-lim,vmax=lim) if title.startswith('Signed') else dict(cmap='Reds',vmin=0,vmax=1))
    im=ax.imshow(img,**kwargs);fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
   ax.set_title(title,fontsize=11);ax.axis('off')
  fig.text(.04,.035,'RGB display clipped to [0,1]; metrics use raw values. Negative width (blue) marks reversed bounds. Heatmap scales vary by sample.',fontsize=10)
  fig.tight_layout(rect=[0,.06,1,.94]);pdf.savefig(fig);fig.savefig(root/f'sample_{i+1:02d}.png',dpi=130);plt.close(fig)
print(root/'uniform_quantile_comparison.pdf')
