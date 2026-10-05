"""Aligned measurements, targets, reconstructions, interval widths and errors."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

METHODS=[('fixed','Fixed quantile head'),('conditioned','Q-conditioned'),('ensemble','Deep ensemble'),('mc','MC dropout')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=8);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    meta=json.loads((a.input.parent/'metadata.json').read_text());data=np.load(a.input)
    n=min(a.limit,len(data['ids']));target=data['target'][:n];measurement=data['measurement'][:n]
    width={k:(data[k+'_hi'][:n]-data[k+'_lo'][:n]).mean(1) for k,_ in METHODS}
    error={k:np.abs(data[k+'_mean'][:n]-target).mean(1) for k,_ in METHODS}
    vmax=max(float(np.quantile(np.stack(list(width.values())),.995)),1e-6)
    emax=max(float(np.quantile(np.stack(list(error.values())),.995)),1e-6)
    coverage_target=meta['coverage_target']
    epochs={meta['methods'][k].get('selected_epoch') for k,_ in METHODS}
    epoch=next(iter(epochs)) if len(epochs)==1 else None
    rows=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    with PdfPages(a.out/'uncertainty_comparison.pdf') as pdf:
        for i in range(n):
            fig,axes=plt.subplots(4,5,figsize=(17,13))
            fig.subplots_adjust(left=.11,right=.975,top=.89,bottom=.13,wspace=.10,hspace=.30)
            fig.suptitle(f"{f'Epoch {epoch} | ' if epoch is not None else ''}Reconstruction and uncertainty | example {i+1} | image {int(data['ids'][i])}",fontsize=20,weight='bold',y=.975)
            fig.text(.11,.935,f"Same input and target for every method | {coverage_target:.0%} calibration target | independent calibration images",fontsize=11,color='#415164')
            for row,(key,label) in enumerate(METHODS):
                pred=data[key+'_mean'][i];lo=data[key+'_lo'][i];hi=data[key+'_hi'][i]
                assert np.all(lo<=hi)
                mse=np.mean((pred.astype(np.float64)-target[i])**2)
                quality=float(ssim(torch.from_numpy(pred[None].copy()),torch.from_numpy(target[i:i+1].copy()),data_range=1.).item())
                cov=float(((lo<=target[i])&(target[i]<=hi)).mean());mean_width=float((hi-lo).mean())
                rows.append({'example':i+1,'id':int(data['ids'][i]),'method':key,'psnr_db':float(-10*np.log10(mse)),'ssim':quality,'coverage':cov,'width':mean_width})
                values=[measurement[i],target[i],pred,width[key][i],error[key][i]]
                for col,v in enumerate(values):
                    ax=axes[row,col];ax.set_axis_off()
                    if col<3:ax.imshow(v.transpose(1,2,0).clip(0,1))
                    else:
                        im=ax.imshow(v,cmap='magma',vmin=0,vmax=vmax if col==3 else emax)
                        if row==3:
                            bar=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.03)
                            bar.ax.tick_params(labelsize=8)
                    if row==0:ax.set_title(['Measurement','Ground truth','Prediction','Uncertainty: interval width','Absolute reconstruction error'][col],fontsize=10,pad=24)
                axes[row,0].text(-.06,.5,label.replace(' ','\n',1),transform=axes[row,0].transAxes,ha='right',va='center',fontsize=11,weight='bold')
                axes[row,2].text(.5,1.015,f"PSNR {rows[-1]['psnr_db']:.2f} dB | SSIM {quality:.3f}",transform=axes[row,2].transAxes,ha='center',fontsize=9)
                axes[row,3].text(.5,1.015,f'Width {mean_width:.4f} | coverage {cov:.1%}',transform=axes[row,3].transAxes,ha='center',fontsize=9)
            fig.text(.11,.082,'Brighter uncertainty = wider pixelwise bounds. Compare with the error map to see whether uncertainty follows actual mistakes.',fontsize=11)
            fig.text(.11,.060,f'Shared scales across all methods and examples: width 0–{vmax:.3f}; error 0–{emax:.3f}. Values above the 99.5th percentile saturate.',fontsize=10,color='#415164')
            fig.text(.11,.039,'Maps average RGB channels. Display images are clipped to [0,1]; metrics are not. The target coverage is not a guarantee for each image.',fontsize=10,color='#415164')
            pdf.savefig(fig);fig.savefig(a.out/f'example_{i+1:02d}_id_{int(data["ids"][i])}.png',dpi=170);plt.close(fig)
    (a.out/'per_example_metrics.json').write_text(json.dumps(rows,indent=2))
    (a.out/'README.txt').write_text(f'{n} examples: first test entries, no visual cherry-picking.\nFour methods with measurement, target, prediction, interval-width and absolute-error maps.\nUncertainty is the RGB-average calibrated interval width; brighter means wider.\nTarget coverage {coverage_target:.0%}; metrics and color scales shared consistently.\n')
    print(a.out/'uncertainty_comparison.pdf')

if __name__=='__main__':main()
