"""Render aligned individual-image reconstruction/width comparisons locally."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchmetrics.functional.image import structural_similarity_index_measure

OUT=Path.home()/'Downloads/ConvRML_Four_Model_Examples'
METHODS=[('fixed','im2im-style fixed head'),('conditioned','Q-conditioned ConvRML'),('ensemble','Deep ensemble'),('mc','MC dropout')]

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.out
    metadata=json.loads((OUT/'metadata.json').read_text())
    target=f"{metadata['coverage_target']:.0%}"
    torch.set_num_threads(2)
    a=np.load(OUT/'examples.npz');x=a['target'];rows=[]
    widths={k:(a[k+'_hi']-a[k+'_lo']).mean(1) for k,_ in METHODS}
    errors={k:np.abs(a[k+'_mean']-x).mean(1) for k,_ in METHODS}
    wmax=float(np.quantile(np.stack(list(widths.values())),.995))
    emax=float(np.quantile(np.stack(list(errors.values())),.995))
    wmax=max(wmax,.001);emax=max(emax,.001)
    for i,image_id in enumerate(a['ids']):
        for k,label in METHODS:
            mean,lo,hi=[a[k+'_'+v][i] for v in ('mean','lo','hi')]
            assert np.all(lo<=hi)
            mse=float(np.mean((mean-x[i])**2))
            ssim=structural_similarity_index_measure(torch.from_numpy(mean[None]),torch.from_numpy(x[i:i+1]),data_range=1.).item()
            rows.append({'example':i+1,'image_id':int(image_id),'key':k,'method':label,'psnr':float(-10*np.log10(mse)),'ssim':ssim,'coverage':float(((lo<=x[i])&(x[i]<=hi)).mean()),'width':float((hi-lo).mean())})
    (OUT/'per_image_metrics.json').write_text(json.dumps(rows,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    with PdfPages(OUT/'four_model_comparison.pdf') as pdf:
        for i,image_id in enumerate(a['ids']):
            fig=plt.figure(figsize=(17,13),facecolor='white')
            gs=fig.add_gridspec(5,5,left=.12,right=.96,top=.91,bottom=.10,hspace=.38,wspace=.14)
            fig.suptitle(f'Same-image comparison • example {i+1} • camera ID {int(image_id)}',fontsize=21,y=.972,weight='bold')
            fig.text(.12,.935,f'Four models • independently calibrated to a {target} average coverage target • identical measurement and ground truth',fontsize=11,color='#455165')
            for j,(title,v) in enumerate([('Input measurement',a['measurement'][i]),('Ground truth',x[i])]):
                ax=fig.add_subplot(gs[0,j]);ax.imshow(v.transpose(1,2,0).clip(0,1));ax.axis('off');ax.set_title(title,fontsize=12)
            ax=fig.add_subplot(gs[0,2:]);ax.axis('off')
            table_rows=[]
            for row in rows[i*4:(i+1)*4]:table_rows.append([row['method'],f"{row['psnr']:.2f}",f"{row['ssim']:.3f}",f"{row['coverage']:.1%}",f"{row['width']:.5f}"])
            table=ax.table(cellText=table_rows,colLabels=['Model','PSNR ↑','SSIM ↑','Coverage','Width ↓'],loc='center',cellLoc='center',colWidths=[.40,.14,.14,.16,.16])
            table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.7)
            ax.set_title('Metrics for this image only',fontsize=12)
            for row,(k,label) in enumerate(METHODS,1):
                images=[a[k+'_mean'][i],errors[k][i],widths[k][i],a[k+'_lo'][i],a[k+'_hi'][i]]
                for col,(title,v) in enumerate(zip(['Reconstruction','Absolute error','Calibrated interval width','Lower pixelwise bound','Upper pixelwise bound'],images)):
                    ax=fig.add_subplot(gs[row,col]);ax.axis('off')
                    if col==0:
                        rowlabel={'fixed':'im2im-style\nfixed head','conditioned':'Q-conditioned\nConvRML','ensemble':'Deep\nensemble','mc':'MC\ndropout'}[k]
                        ax.text(-.08,.5,rowlabel,transform=ax.transAxes,va='center',ha='right',fontsize=10,weight='bold')
                    if v.ndim==3:ax.imshow(v.transpose(1,2,0).clip(0,1))
                    else:
                        im=ax.imshow(v,cmap='magma',vmin=0,vmax=emax if col==1 else wmax)
                        if row==4:
                            bar=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.025)
                            bar.ax.tick_params(labelsize=8)
                    if row==1:ax.set_title(title,fontsize=11)
            fig.text(.12,.063,f'Heatmap scales are shared across all methods and all eight examples: error 0–{emax:.3f}; width 0–{wmax:.3f}. Larger values saturate.',fontsize=10,color='#455165')
            fig.text(.12,.043,'RGB images are clipped to [0,1] for display only; metrics use unclipped values. Bounds are pixelwise endpoints, not sampled reconstructions.',fontsize=10,color='#455165')
            fig.text(.12,.023,f'{target} is an average calibration target, not guaranteed coverage for this individual image. The models have different training procedures and budgets.',fontsize=10,color='#455165')
            pdf.savefig(fig)
            fig.savefig(OUT/f'example_{i+1:02d}_id_{int(image_id)}.png',dpi=170,facecolor='white')
            plt.close(fig)
    # Compact width chart for the first image, alongside its actual coverage.
    fig,ax=plt.subplots(figsize=(10,5.5));fig.subplots_adjust(left=.30,right=.96,bottom=.20,top=.80)
    first=rows[:4];ys=np.arange(4);vals=[r['width'] for r in first]
    ax.barh(ys,vals,color=['#5898b0','#197c80','#8293b9','#c7a274']);ax.set_yticks(ys,[r['method'] for r in first]);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.65)
    for j,r in enumerate(first):ax.text(r['width']+max(vals)*.025,j,f"{r['width']:.5f} | {r['coverage']:.1%} covered",va='center',fontsize=10)
    ax.set_xlabel('Mean calibrated interval width');ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Interval width on the same image • camera ID {int(a["ids"][0])}',fontsize=16,weight='bold')
    fig.text(.08,.08,f'Image-specific coverage shown beside each bar; target calibration coverage is {target} on average.',fontsize=10)
    fig.savefig(OUT/'first_image_width_chart.png',dpi=220,facecolor='white');plt.close(fig)
    (OUT/'README.txt').write_text(f'Eight examples selected as the first eight entries of the shared 2500-image real-RML test manifest, also included in the 1000-image evaluation.\nAll four methods use their completed checkpoints and held-out empirical {target} calibration.\nThe fixed-head method is a ConvRML adaptation of im2im pixelwise quantile regression.\nNo new training or noise simulation. Source values are stored in examples.npz; per-image metrics in per_image_metrics.json.\n')
    print(OUT/'four_model_comparison.pdf')

if __name__=='__main__':main()
