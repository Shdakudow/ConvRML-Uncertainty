"""Learn pooled signed residual quantiles; test pixel marginals without retraining."""
import csv,hashlib,json,sys
from pathlib import Path
from statistics import NormalDist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
ROOT=Path('results/residual_distribution_20260925')
SOURCE=Path('results/sampling_pilot_calibration95_20260925')
TD=Path('data/formal_rml_25k');CD=Path('data/formal_rml_calibration95')
LEVELS=np.array([.005,.01,.025,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.975,.99,.995])
NORMAL=np.array([NormalDist().inv_cdf(float(q)) for q in LEVELS])
VARIANTS=['empirical_residual','gaussian_residual','gaussian_matching_calibrated95']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def arrays(method,role):return {k:np.load(SOURCE/method/f'{role}_{k}.npy',mmap_mode='r') for k in ['r','lo','hi']}
def widths(r,lo,hi):return np.maximum(r-lo,.001),np.maximum(hi-r,.001)
def transform(r,wl,wu,z):return r+np.where(np.asarray(z)<0,wl,wu)*z
def quantile(variant,r,wl,wu,z,factor):
    if variant!='gaussian_matching_calibrated95':return transform(r,wl,wu,z)
    # Fit a literal symmetric Gaussian to the existing calibrated L/U.
    mu=r+factor*(wu-wl)/2
    sigma=factor*(wl+wu)/(2*NormalDist().inv_cdf(.975))
    return mu+sigma*z

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((SOURCE/'inference.json').read_text());cal=json.loads((SOURCE/'calibration.json').read_text())
    assert metadata['test_manifest_sha256']==sha(TD/'manifest.json')
    assert metadata['calibration_manifest_sha256']==sha(CD/'manifest.json')
    cp=np.load(CD/'pairs.npy',mmap_mode='r');tp=np.load(TD/'pairs.npy',mmap_mode='r')
    manifest=json.loads((TD/'manifest.json').read_text());fits={};records=[];results={}
    for method in ['uniform','endpoints']:
        a=arrays(method,'calibration')
        store=np.lib.format.open_memmap(ROOT/f'{method}_normalized_residuals.npy',mode='w+',dtype=np.float32,shape=(2048,3,256,256))
        total=0.;square=0.;count=0
        for s in range(0,2048,8):
            r,lo,hi=[a[k][s:s+8] for k in ['r','lo','hi']];x=cp[s:s+8,1].astype(np.float32)/255
            wl,wu=widths(r,lo,hi);error=x-r;z=error/np.where(error<0,wl,wu)
            assert np.isfinite(z).all();store[s:s+8]=z
            total+=z.sum(dtype=np.float64);square+=np.square(z.astype(np.float64)).sum();count+=z.size
        store.flush();mu=total/count;std=max(0.,square/count-mu*mu)**.5
        zgrid=np.quantile(store,LEVELS,method='linear',overwrite_input=True);del store
        assert np.all(np.diff(zgrid)>=0)
        fits[method]={'levels':LEVELS.tolist(),'empirical_z':zgrid.tolist(),'gaussian_z_mean':mu,'gaussian_z_std':std,'count':count,'images':2048}
        print('FITTED',method,json.dumps(fits[method]),flush=True)
    # Freeze the learned distributions before test evaluation.
    (ROOT/'distribution_fits.json').write_text(json.dumps(fits,indent=2))
    for method,fit in fits.items():
        a=arrays(method,'test');factor=cal['factors'][method]
        zg={'empirical_residual':np.array(fit['empirical_z']),
            'gaussian_residual':fit['gaussian_z_mean']+fit['gaussian_z_std']*NORMAL,
            'gaussian_matching_calibrated95':NORMAL}
        hit={v:np.zeros(len(LEVELS)) for v in VARIANTS};loss={v:np.zeros(len(LEVELS)) for v in VARIANTS};samplecount=0
        per={v:[] for v in VARIANTS}
        # 4096 uniformly sampled pixel-channel locations per test image,
        # identical locations for every method and variant.
        rng=np.random.default_rng(20260925)
        for s in range(0,1000,8):
            e=min(s+8,1000);n=e-s;r,lo,hi=[a[k][s:e] for k in ['r','lo','hi']]
            x=tp[22500+s:22500+e,1].astype(np.float32)/255;wl,wu=widths(r,lo,hi)
            index=rng.integers(0,3*256*256,size=(n,4096))
            rs,wls,wus,xs=[np.take_along_axis(v.reshape(n,-1),index,axis=1) for v in [r,wl,wu,x]]
            samplecount+=n*4096
            for variant in VARIANTS:
                lower=quantile(variant,r,wl,wu,zg[variant][2],factor)
                upper=quantile(variant,r,wl,wu,zg[variant][-3],factor)
                assert np.all(lower<=upper)
                coverage=((lower<=x)&(x<=upper)).mean((1,2,3));width=(upper-lower).mean((1,2,3))
                image_loss=np.zeros(n)
                for j,q in enumerate(LEVELS):
                    pred=quantile(variant,rs,wls,wus,zg[variant][j],factor)
                    error=xs-pred;pb=np.maximum(q*error,(q-1)*error)
                    hit[variant][j]+=(xs<=pred).sum();loss[variant][j]+=pb.sum(dtype=np.float64)
                    image_loss+=pb.mean(1)/len(LEVELS)
                for i in range(n):
                    row={'model':method,'distribution':variant,'id':manifest['test'][s+i]['id'],'coverage95':float(coverage[i]),'width95':float(width[i]),'grid_mean_pinball':float(image_loss[i])}
                    records.append(row);per[variant].append(row)
            if s%200==0:print('EVALUATED',method,e,flush=True)
        results[method]={}
        for variant in VARIANTS:
            observed=hit[variant]/samplecount
            results[method][variant]={**{key:float(np.mean([row[key] for row in per[variant]])) for key in ['coverage95','width95','grid_mean_pinball']},
                'quantile_mean_absolute_calibration_error':float(np.abs(observed-LEVELS).mean()),
                'observed_cdf':observed.tolist(),'pinball_by_quantile':(loss[variant]/samplecount).tolist()}
    with (ROOT/'per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    payload={'fits':fits,'results':results,'levels':LEVELS.tolist(),'calibration_count':2048,'test_count':1000,
             'evaluation':'95% interval metrics use all pixel-channels; quantile curves and mean grid pinball use 4096 pixel-channel samples/image with replacement; same draws for all methods',
             'scope':'Pooled marginal residual-shape approximation, not per-pixel conditional identification or joint-image distribution. No neural retraining; reused held-out test set, no new test-based fitting. Interior quantiles interpolate a learned grid; tails beyond .005/.995 are not characterized.'}
    (ROOT/'comparison.json').write_text(json.dumps(payload,indent=2))
    lines=['# Pixel uncertainty from normalized residuals','',payload['scope'],'',payload['evaluation'],'',
           '| Model | Distribution | 95% coverage | Mean width | Grid pinball | Quantile calibration MAE |',
           '|---|---|---:|---:|---:|---:|']
    for method,variants in results.items():
        for variant,r in variants.items():lines.append(f"| {method} | {variant} | {r['coverage95']:.2%} | {r['width95']:.6f} | {r['grid_mean_pinball']:.6f} | {r['quantile_mean_absolute_calibration_error']:.4f} |")
    lines+=['','Empirical residual: pooled signed z=(x-r)/side_width, with widths floored at .001; learn quantiles of z on calibration only. Map each z quantile through the pixel-specific positive lower/upper scale. Gaussian residual: fit mean/std to the same z values, then use the identical scale transform. Gaussian matching calibrated95: a literal Gaussian whose 2.5/97.5 percentiles match the existing calibrated endpoints; its center may differ from r. Reconstruction outputs are unchanged for all variants. Grid mean pinball is an equal-weight average over the specified nonuniform quantile grid, not CRPS. Quantile calibration MAE is pooled and may hide spatial/subgroup miscalibration. Narrower widths alone are not improvements if coverage falls.']
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
    draw(payload);print('\n'.join(lines),flush=True)

def draw(payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for col,method in enumerate(['uniform','endpoints']):
        for variant,r in payload['results'][method].items():
            axes[0,col].plot(LEVELS,r['observed_cdf'],marker='.',label=variant.replace('_',' '))
            axes[1,col].plot(LEVELS,r['pinball_by_quantile'],marker='.',label=variant.replace('_',' '))
        axes[0,col].plot([0,1],[0,1],'k--',label='ideal calibration')
        axes[0,col].set(title=method+' model: quantile calibration',xlabel='Requested quantile',ylabel='Fraction of targets below prediction')
        axes[1,col].set(title='Quantile prediction loss',xlabel='Quantile',ylabel='Mean pinball loss')
        axes[0,col].legend(fontsize=7)
    fig.suptitle('Residual distribution experiment | 2048 calibration / 1000 test images')
    fig.savefig(ROOT/'distribution_comparison.png',dpi=170)
    with PdfPages(ROOT/'distribution_comparison.pdf') as pdf:pdf.savefig(fig)
    plt.close(fig)

if __name__=='__main__':main()
