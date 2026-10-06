from pathlib import Path
import json,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import structural_similarity
root=Path('results/toy_diagnosis');d=json.loads((root/'diagnosis.json').read_text());a=np.load(root/'samples.npz');t=d['test'];b=d['batch_stats_first128'];e=d['eval_first128']
def page_text(pdf,title,paragraphs):
 fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.91,title,fontsize=22,weight='bold');y=.82
 for p in paragraphs:
  wrapped=textwrap.fill(p,105);fig.text(.07,y,wrapped,fontsize=12,va='top',linespacing=1.5);y-=.038*(wrapped.count('\n')+1)+.035
 pdf.savefig(fig);plt.close(fig)
with PdfPages(root/'toy_reconstruction_diagnosis.pdf') as pdf:
 page_text(pdf,'Toy reconstruction and uncertainty: diagnostic report',[
 'Job 585165 completed 8 epochs in 14 min 34 sec. This report uses the checkpoint selected by lowest validation MSE (epoch %d). No additional training or calibration was performed.'%d['checkpoint_epoch'],
 'Data: parallel_100k/gt2rml targets, resized to 64 x 64. Synthetic measurements use circular convolution with a random anisotropic Gaussian PSF, a central 48 x 48 crop, and Gaussian noise. These are not real DiffuserCam measurements; the source targets are aligned to RML.',
 'Held-out test: 1,024 images, source offsets 9,216–10,239, disjoint from training and validation. The 10 displayed examples are the first 10 test images, selected without inspecting results.',
 'Test reconstruction: PSNR %.2f dB; SSIM %.3f. PSNR uses aggregate MSE over RGB pixels; all numerical metrics use raw predictions, not display clipping.'%(t['psnr'],t['ssim']),
 'Uncalibrated bounds: P(x <= lower) = %.1f%% (target 5%%); P(x <= upper) = %.1f%% (target 95%%). Actual interval coverage = %.1f%%; crossed RGB entries = %.1f%%.'%(100*t['lower_cdf'],100*t['upper_cdf'],100*t['interval_coverage'],100*t['crossing']),
 'The bounds are not valid 90% intervals. The paper\'s held-out risk calibration has not been implemented. Reconstruction quality and interval reliability must be assessed separately.'
 ])
 page_text(pdf,'Diagnosis: quantile-head normalization',[
 'The same quantile head contains four BatchNorm layers. Each training batch runs the lower endpoint and then the upper endpoint separately, using different batch statistics. Evaluation uses a single shared running mean and variance for both endpoints.',
 'Controlled ablation: same checkpoint and first 128 test images; encoder and reconstruction head remain in evaluation mode. Only quantile-head BatchNorm switches to current-batch statistics. No gradient updates are performed.',
 'Normal inference: endpoint CDFs %.1f%% / %.1f%%; coverage %.1f%%; crossing %.1f%%.'%(100*e['lower_cdf'],100*e['upper_cdf'],100*e['interval_coverage'],100*e['crossing']),
 'Batch-statistics diagnostic: endpoint CDFs %.1f%% / %.1f%%; coverage %.1f%%; crossing %.1f%%.'%(100*b['lower_cdf'],100*b['upper_cdf'],100*b['interval_coverage'],100*b['crossing']),
 'A strong improvement in this ablation implicates shared BatchNorm statistics as a major contributor. It does not establish calibrated coverage or justify using test-batch statistics in deployment.',
 'Recommended repair: replace normalization in the quantile branch with a batch-independent design (or maintain endpoint-specific statistics), retrain, then validate ordering and coverage. Keep the normal reconstruction head intact. Reserve a separate calibration split before final test evaluation.',
 'Additional limitations: synthetic PSF is not supplied to the model; image resolution is low and dark backgrounds influence whole-image metrics; results cannot be compared directly with the real-measurement advisor baseline. Reference: Angelopoulos et al., arXiv:2202.05265.'
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
print(root/'toy_reconstruction_diagnosis.pdf')
