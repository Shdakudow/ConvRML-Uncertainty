from pathlib import Path
import json,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
base=Path('results'); dest=Path('/Users/shdakudow/Downloads/ConvRML_Toy_Report');dest.mkdir(parents=True,exist_ok=True)
reports={k:json.loads((base/f'motion_{k}'/'diagnosis.json').read_text()) for k in ['endpoint','uniform']}
data={k:np.load(base/f'motion_{k}'/'samples.npz') for k in reports}
time={k:int((base/f'motion_{k}'/'training_seconds.txt').read_text()) for k in reports}
assert np.array_equal(data['endpoint']['x'],data['uniform']['x'])
assert np.array_equal(data['endpoint']['y'],data['uniform']['y'])
def textpage(pdf,title,paras):
 f=plt.figure(figsize=(11.7,8.3));f.text(.06,.92,title,fontsize=21,weight='bold');y=.84
 for p in paras:
  p=textwrap.fill(p,110);f.text(.06,y,p,fontsize=11,va='top',linespacing=1.4);y-=.03*(p.count('\n')+1)+.03
 pdf.savefig(f);plt.close(f)
with PdfPages(dest/'motion_psf_comparison.pdf') as pdf:
 paras=['Controlled motion-blur comparison, job 586876. Both models use the same ConvNeXt-Tiny encoder and normal MSE reconstruction head. GroupNorm replaces BatchNorm only in the quantile head. Both use MSE plus pinball loss, with no crossing penalty.',
 'Endpoint training averages pinball losses at .05 and .95. Uniform training samples one independent u from Uniform(0,1) per image per step. Both run eight epochs on 8192 images at batch32, from the same initialization seed and data order. Sequential execution on the same RTX6000Ada allocation; timing includes training, validation, and checkpoint writes, but excludes final test inference.',
 'Fresh disjoint source slices: train [12000,20192), validation [20192,21216), test [21216,22240). Source: parallel_100k/gt2rml. 64x64 images; circular convolution with a random line PSF (length 5–17px, angle 0–180 degrees, transverse Gaussian sigma .5px), central 48x48 crop, noise sigma .01–.05. PSF is not given to the model.',
 'The checkpoint with lowest validation MSE is evaluated on 1024 test images. Ten displayed examples are the first ten, selected before seeing results. All numerical results use raw predictions; RGB displays are clipped to [0,1].']
 for k in reports:
  t=reports[k]['test'];paras.append('%s: best epoch %d; training %dm%02ds; PSNR %.2f dB; SSIM %.3f; coverage %.2f%%; crossed RGB entries %.3f%%; endpoint CDFs %.2f%% / %.2f%%; mean signed width %.4f.'%(k.title(),reports[k]['checkpoint_epoch'],time[k]//60,time[k]%60,t['psnr'],t['ssim'],100*t['interval_coverage'],100*t['crossing'],100*t['lower_cdf'],100*t['upper_cdf'],t['mean_signed_width']))
 textpage(pdf,'Motion PSF: endpoint versus uniform training',paras)
 e,u=[reports[k]['test'] for k in ['endpoint','uniform']]
 textpage(pdf,'How to interpret this comparison',[
 'Uniform minus endpoint: PSNR %+.3f dB; SSIM %+.4f; coverage %+.2f percentage points; crossings %+.3f percentage points. Uniform training runtime / endpoint runtime = %.3f.'%(u['psnr']-e['psnr'],u['ssim']-e['ssim'],100*(u['interval_coverage']-e['interval_coverage']),100*(u['crossing']-e['crossing']),time['uniform']/time['endpoint']),
 'Higher PSNR/SSIM indicate better point reconstruction. Desired interval coverage is 90%; endpoint CDFs should approach 5% and 95%. Crossings should approach zero. Higher coverage alone is not always better: excessively wide intervals are uninformative. Signed width is descriptive and can cancel where bounds cross.',
 'This comparison removes the earlier BatchNorm mismatch and crossing-penalty difference. It is one run per method, not evidence of a universal ranking. Different objectives consume randomness differently during training even with matched initial seeds. Best-checkpoint selection emphasizes reconstruction, not calibration.',
 'The Gaussian comparison favored uniform uncertainty but used the flawed BatchNorm endpoint baseline. Motion results must be interpreted within this repaired pair; differences from the Gaussian runs also include normalization and image splits.',
 'No held-out risk calibration or distribution-free guarantee is claimed. Dark backgrounds contribute strongly to whole-image metrics. This is a low-resolution synthetic inverse problem, not measured DiffuserCam performance.',
 'Each sample page compares both methods on exactly the same input. Signed-width maps share one color scale between methods within each sample; blue denotes reversed bounds. Error maps also share a scale. Uncertainty and error need not match at every pixel.'
 ])
 for i in range(10):
  arr={k:{v:data[k][v][i].transpose(1,2,0) for v in ['y','x','r','lo','hi']} for k in data}
  e,u=arr['endpoint'],arr['uniform'];we=(e['hi']-e['lo']).mean(-1);wu=(u['hi']-u['lo']).mean(-1);ee=np.abs(e['r']-e['x']).mean(-1);eu=np.abs(u['r']-u['x']).mean(-1)
  ims=[e['y'],e['x'],e['r'],u['r'],e['lo'],e['hi'],u['lo'],u['hi'],ee,eu,we,wu]
  titles=['Synthetic input','Ground truth','Endpoint reconstruction','Uniform reconstruction','Endpoint lower .05','Endpoint upper .95','Uniform lower .05','Uniform upper .95','Endpoint absolute error','Uniform absolute error','Endpoint signed width','Uniform signed width']
  f,axs=plt.subplots(3,4,figsize=(14,10));f.suptitle('Test %02d | %s'%(i+1,data['endpoint']['names'][i]),fontsize=14)
  for j,(ax,img,title) in enumerate(zip(axs.flat,ims,titles)):
   if j<8:ax.imshow(np.clip(img,0,1))
   else:
    lim=max(float(max(np.abs(we).max(),np.abs(wu).max())),1e-6) if j>=10 else max(float(max(ee.max(),eu.max())),1e-6)
    m=ax.imshow(img,cmap='coolwarm' if j>=10 else 'magma',vmin=-lim if j>=10 else 0,vmax=lim);f.colorbar(m,ax=ax,fraction=.045)
   ax.set_title(title,fontsize=10);ax.axis('off')
  f.tight_layout(rect=[0,.02,1,.96]);pdf.savefig(f);plt.close(f)
print(dest/'motion_psf_comparison.pdf')
log=dest/'EXPERIMENT_LOG.md'
with log.open('a') as h:
 h.write('\n## Motion results — job 586876\n\n')
 for k in reports:h.write(f'- {k}: {time[k]} seconds training/validation/checkpointing; '+json.dumps(reports[k])+'\n')
 h.write('\nPDF: motion_psf_comparison.pdf. No calibration or further training performed.\n')
