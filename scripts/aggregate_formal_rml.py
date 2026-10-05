"""Aggregate five aligned members by total variance, compare with MC dropout."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
ROOT=Path('results/formal_rml_25k');DATA=Path('data/formal_rml_25k')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/formal_rml_25k');args=p.parse_args()
    global ROOT
    ROOT=Path(args.root)
    manifest_hash=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
    reports=[json.loads((ROOT/f'worker_{i}/complete.json').read_text()) for i in range(6)]
    assert all(r['manifest_sha256']==manifest_hash and r['count']==2500 for r in reports)
    assert [r['worker'] for r in reports]==list(range(6))
    means=[np.load(ROOT/f'worker_{i}/mean.npy',mmap_mode='r') for i in range(6)]
    variances=[np.load(ROOT/f'worker_{i}/variance.npy',mmap_mode='r') for i in range(6)]
    pairs=np.load(DATA/'pairs.npy',mmap_mode='r');sums={k:np.zeros(5) for k in ('ensemble','mc')};examples={}
    for start in range(0,2500,8):
        end=min(start+8,2500);x=pairs[22500+start:22500+end,1].astype(np.float32)/255
        m=np.stack([v[start:end] for v in means[:5]]);v=np.stack([a[start:end] for a in variances[:5]])
        predictions={'ensemble':(m.mean(0),v.mean(0)+m.var(0)),'mc':(means[5][start:end],variances[5][start:end])}
        for name,(mu,var) in predictions.items():
            if not np.isfinite(mu).all() or not np.isfinite(var).all() or (var<0).any():raise ValueError('Invalid saved predictions')
            radius=1.6448536269514722*np.sqrt(var);lo,hi=mu-radius,mu+radius
            sums[name]+=np.array([np.mean((mu-x)**2),np.mean((lo<=x)&(x<=hi)),np.mean(hi-lo),np.mean(x<lo),np.mean(x>hi)])*(end-start)
            if start==0:examples[name]={'mean':mu.copy(),'lo':lo,'hi':hi}
    result={name:dict(zip(('mse','coverage','width','lower_miss','upper_miss'),(v/2500).tolist())) for name,v in sums.items()}
    for r in result.values():r['psnr']=-10*np.log10(r['mse'])
    report={'test_count':2500,'manifest_sha256':manifest_hash,'members':reports,'metrics':result,'calibration':False,'notes':['Gaussian moment intervals, nominal 90%; no guaranteed coverage.','Ensemble total variance includes learned aleatoric plus between-member variance.','MC intervals use dropout variance only; no unsupported weight-decay precision term.','Same manifest and sample order verified before aggregation.']}
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(ROOT/'comparison.pdf') as pdf:
        for i in range(8):
            fig,axes=plt.subplots(2,4,figsize=(14,7))
            for row,name in enumerate(('ensemble','mc')):
                e=examples[name]
                for j,(title,a) in enumerate([('Target',pairs[22500+i,1]/255),('Mean',e['mean'][i]),('Interval width',(e['hi'][i]-e['lo'][i]).mean(0)),('Absolute error',np.abs(e['mean'][i]-pairs[22500+i,1]/255).mean(0))]):
                    ax=axes[row,j];ax.axis('off');ax.set_title(name+': '+title)
                    if a.ndim==3:ax.imshow(a.transpose(1,2,0).clip(0,1))
                    else:ax.imshow(a,cmap='magma',vmin=0,vmax=.5 if j==2 else .2)
            fig.suptitle(f'Real RML test example {i+1}; uncalibrated Gaussian intervals')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)

if __name__=='__main__':main()
