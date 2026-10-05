"""Frozen-model local-context residual CDFs with image-disjoint fitting/calibration."""
import json,csv,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import uniform_filter,sobel
from scripts.residual_distribution_experiment import arrays,widths,transform,LEVELS,TD,CD,SOURCE,sha
ROOT=Path('results/context_residual_20260925')
GROUPS=['smooth','intermediate','high_structure']

def context(r):
    gray=r.mean(1)
    mean=uniform_filter(gray,size=(1,5,5),mode='reflect')
    variance=np.maximum(uniform_filter(gray*gray,size=(1,5,5),mode='reflect')-mean*mean,0)
    # Apply Sobel within each image, never across the batch dimension.
    gx=np.stack([sobel(v,axis=1,mode='reflect')/8 for v in gray])
    gy=np.stack([sobel(v,axis=0,mode='reflect')/8 for v in gray])
    return np.sqrt(variance)+uniform_filter(np.hypot(gx,gy),size=(1,5,5),mode='reflect')

def collect(method,start,count,seed):
    a=arrays(method,'calibration');pairs=np.load(CD/'pairs.npy',mmap_mode='r')
    fields={k:[] for k in ['r','wl','wu','x','context']};rng=np.random.default_rng(seed)
    for s in range(start,start+count,8):
        e=min(s+8,start+count);r,lo,hi=[a[k][s:e] for k in ['r','lo','hi']];wl,wu=widths(r,lo,hi)
        x=pairs[s:e,1].astype(np.float32)/255;f=context(r)
        idx=rng.integers(0,3*256*256,size=(e-s,4096))
        for key,v in zip(['r','wl','wu','x'],[r,wl,wu,x]):fields[key].append(np.take_along_axis(v.reshape(e-s,-1),idx,axis=1))
        fields['context'].append(np.take_along_axis(f.reshape(e-s,-1),idx%(256*256),axis=1))
    return {k:np.concatenate(v) for k,v in fields.items()}

def fit(fields):
    cuts=np.quantile(fields['context'],[1/3,2/3]);groups=np.searchsorted(cuts,fields['context'],side='left')
    error=fields['x']-fields['r'];z=error/np.where(error<0,fields['wl'],fields['wu'])
    order=np.argsort(z.ravel());zs=z.ravel()[order];gs=groups.ravel()[order];n=len(zs)
    global_q=np.interp(LEVELS,np.arange(1,n+1)/n,zs)
    tables=[];alphas=[];counts=[]
    for group in range(3):
        present=gs==group;ng=int(present.sum());images=int(np.any(groups==group,axis=1).sum())
        alpha=min(.75,images/(images+100)) if ng else 0.
        # A true CDF mixture: alpha*group ECDF+(1-alpha)*global ECDF.
        weights=np.full(n,(1-alpha)/n)
        if ng:weights+=present*(alpha/ng)
        cdf=np.cumsum(weights);cdf/=cdf[-1]
        tables.append(np.interp(LEVELS,cdf,zs));alphas.append(alpha);counts.append({'pixel_samples':ng,'images':images})
    table=np.stack(tables)
    assert np.all(np.diff(table,axis=1)>=0) and np.all(np.diff(global_q)>=0)
    assert np.all(table[:,2]<0) and np.all(table[:,-3]>0) and global_q[2]<0 and global_q[-3]>0
    return {'cuts':cuts.tolist(),'global':global_q.tolist(),'context':table.tolist(),'mixture_weights':alphas,'group_counts':counts}

def quantiles(fit,variant,groups):
    if variant=='global':return np.broadcast_to(np.array(fit['global']),groups.shape+(len(LEVELS),))
    return np.array(fit['context'])[groups]

def sampled_metrics(fields,fit,variant,factor=1.):
    groups=np.searchsorted(fit['cuts'],fields['context'],side='left')
    hit=np.zeros((3,len(LEVELS)));loss=np.zeros_like(hit);counts=np.bincount(groups.ravel(),minlength=3)
    image_pb=np.empty((len(fields['r']),len(LEVELS)))
    for j,q in enumerate(LEVELS):
        z=np.array(fit[variant])[groups,j] if variant=='context' else fit['global'][j]
        pred=transform(fields['r'],fields['wl'],fields['wu'],factor*z)
        err=fields['x']-pred;pb=np.maximum(q*err,(q-1)*err);image_pb[:,j]=pb.mean(1)
        for g in range(3):
            mask=groups==g;hit[g,j]=((fields['x']<=pred)&mask).sum();loss[g,j]=pb[mask].sum(dtype=np.float64)
    return hit,loss,counts,image_pb

def finalize(hit,loss,counts,pb):
    observed=hit.sum(0)/counts.sum();average_loss=loss.sum(0)/counts.sum()
    return {'grid_mean_pinball':float(pb.mean()),'truncated_integrated_quantile_score':float(np.mean(2*np.trapz(pb,LEVELS,axis=1))),
            'quantile_calibration_mae':float(np.abs(observed-LEVELS).mean()),'observed_cdf':observed.tolist(),
            'groups':{GROUPS[g]:{'sample_count':int(counts[g]),'quantile_calibration_mae':float(np.abs(hit[g]/counts[g]-LEVELS).mean()),
                                  'grid_mean_pinball':float(np.mean(loss[g]/counts[g]))} if counts[g] else {'sample_count':0} for g in range(3)}}

def calibrate(method,fit,variant):
    a=arrays(method,'calibration');pairs=np.load(CD/'pairs.npy',mmap_mode='r')
    scores=np.lib.format.open_memmap(ROOT/f'{method}_{variant}_scores.npy',mode='w+',dtype=np.float32,shape=(512,3,256,256))
    for s in range(1536,2048,8):
        r,lo,hi=[a[k][s:s+8] for k in ['r','lo','hi']];wl,wu=widths(r,lo,hi);x=pairs[s:s+8,1].astype(np.float32)/255
        groups=np.searchsorted(fit['cuts'],context(r),side='left')[:,None]
        zlo=np.array(fit['context'])[groups,2] if variant=='context' else fit['global'][2]
        zhi=np.array(fit['context'])[groups,-3] if variant=='context' else fit['global'][-3]
        score=np.maximum((r-x)/(wl*(-zlo)),(x-r)/(wu*zhi))
        assert np.isfinite(score).all();scores[s-1536:s-1536+8]=score
    scores.flush();factor=max(0.,float(np.quantile(scores,.95,method='higher',overwrite_input=True)));del scores
    return factor

def main():
    ROOT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    info=json.loads((SOURCE/'inference.json').read_text())
    assert info['calibration_manifest_sha256']==sha(CD/'manifest.json') and info['test_manifest_sha256']==sha(TD/'manifest.json')
    fits={};dev={};factors={}
    for method in ['uniform','endpoints']:
        fields=collect(method,0,1024,4201);fits[method]=fit(fields);del fields
        fields=collect(method,1024,512,4202);dev[method]={}
        for variant in ['global','context']:dev[method][variant]=finalize(*sampled_metrics(fields,fits[method],variant))
        del fields
        factors[method]={v:calibrate(method,fits[method],v) for v in ['global','context']}
        print('FITTED',method,'factors',factors[method],flush=True)
    # No choices are made from development outcomes: fixed design and blend rule.
    (ROOT/'frozen_fit.json').write_text(json.dumps({'fits':fits,'factors':factors,'development':dev,'levels':LEVELS.tolist(),
        'split':{'fit_indices':[0,1024],'development_indices':[1024,1536],'calibration_indices':[1536,2048]},'source':info},indent=2))
    tp=np.load(TD/'pairs.npy',mmap_mode='r');manifest=json.loads((TD/'manifest.json').read_text())
    results={};records=[];paired={}
    for method in fits:
        a=arrays(method,'test');fit_=fits[method];rng=np.random.default_rng(4203)
        names=['global_raw','context_raw','global_calibrated','context_calibrated']
        accum={name:{'hit':np.zeros((3,len(LEVELS))),'loss':np.zeros((3,len(LEVELS))),'counts':np.zeros(3,dtype=np.int64),'pb':[],'coverage':[],'width':[],'group_hits':np.zeros(3),'group_total':np.zeros(3),'group_width':np.zeros(3)} for name in names}
        for s in range(0,1000,8):
            e=min(s+8,1000);n=e-s;r,lo,hi=[a[k][s:e] for k in ['r','lo','hi']];wl,wu=widths(r,lo,hi)
            x=tp[22500+s:22500+e,1].astype(np.float32)/255;f=context(r);groups=np.searchsorted(fit_['cuts'],f,side='left')
            index=rng.integers(0,3*256*256,size=(n,4096));fields={key:np.take_along_axis(v.reshape(n,-1),index,axis=1) for key,v in zip(['r','wl','wu','x'],[r,wl,wu,x])}
            fields['context']=np.take_along_axis(f.reshape(n,-1),index%(256*256),axis=1)
            for name in names:
                variant=name.split('_')[0];factor=factors[method][variant] if name.endswith('calibrated') else 1.
                hit,loss,counts,pb=sampled_metrics(fields,fit_,variant,factor);ac=accum[name]
                ac['hit']+=hit;ac['loss']+=loss;ac['counts']+=counts;ac['pb'].append(pb)
                zl=np.array(fit_['context'])[groups[:,None],2] if variant=='context' else fit_['global'][2]
                zh=np.array(fit_['context'])[groups[:,None],-3] if variant=='context' else fit_['global'][-3]
                lower=r+wl*(factor*zl);upper=r+wu*(factor*zh)
                covered=(lower<=x)&(x<=upper);width=upper-lower
                cov=covered.mean((1,2,3));wid=width.mean((1,2,3));ac['coverage'].extend(cov.tolist());ac['width'].extend(wid.tolist())
                for g in range(3):
                    mask=np.broadcast_to(groups[:,None]==g,x.shape);ac['group_total'][g]+=mask.sum();ac['group_hits'][g]+=(covered&mask).sum();ac['group_width'][g]+=width[mask].sum(dtype=np.float64)
                score=2*np.trapz(pb,LEVELS,axis=1)
                for i in range(n):records.append({'model':method,'variant':name,'id':manifest['test'][s+i]['id'],'coverage95':float(cov[i]),'width95':float(wid[i]),'truncated_score':float(score[i]),'grid_pinball':float(pb[i].mean())})
            if s%200==0:print('EVALUATED',method,e,flush=True)
        results[method]={}
        for name,ac in accum.items():
            pb=np.concatenate(ac['pb']);result=finalize(ac['hit'],ac['loss'],ac['counts'],pb)
            result.update(coverage95=float(np.mean(ac['coverage'])),width95=float(np.mean(ac['width'])))
            for g in range(3):
                if ac['group_total'][g]:result['groups'][GROUPS[g]].update(coverage95=float(ac['group_hits'][g]/ac['group_total'][g]),width95=float(ac['group_width'][g]/ac['group_total'][g]))
            results[method][name]=result
        paired[method]={}
        for suffix in ['raw','calibrated']:
            global_scores=2*np.trapz(np.concatenate(accum['global_'+suffix]['pb']),LEVELS,axis=1)
            context_scores=2*np.trapz(np.concatenate(accum['context_'+suffix]['pb']),LEVELS,axis=1)
            delta=context_scores-global_scores;boot=np.random.default_rng(4204).integers(0,len(delta),(1000,len(delta)))
            paired[method][suffix]={'mean_context_minus_global_score':float(delta.mean()),'image_bootstrap_95_interval':np.quantile(delta[boot].mean(1),[.025,.975]).tolist()}
    with (ROOT/'per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    output={'results':results,'paired_score_difference':paired,'seconds':time.monotonic()-start,
            'scope':'Both variants refitted on same 1024 images; 512 development images (no tuning); 512 separate final calibration images; 1000 existing test images. Quantile scores sampled at 4096 RGB pixel-channel locations/image; intervals evaluated at all pixels. Score integrates pinball over .005-.995 with trapezoid quadrature, not full CRPS. Bootstrap resamples images and assumes image independence; near-duplicate scenes could reduce effective sample size.'}
    (ROOT/'comparison.json').write_text(json.dumps(output,indent=2))
    lines=['# Context-dependent residual experiment','',output['scope'],'',
           '| Model | Variant | 95% coverage | Width | Truncated integrated score | Quantile calibration MAE |','|---|---|---:|---:|---:|---:|']
    for method,variants in results.items():
        for variant,v in variants.items():lines.append(f"| {method} | {variant} | {v['coverage95']:.2%} | {v['width95']:.6f} | {v['truncated_integrated_quantile_score']:.6f} | {v['quantile_calibration_mae']:.4f} |")
    lines+=['','Context is 5x5 local reconstruction standard deviation plus local mean Sobel magnitude. Fit-only tertiles create smooth/intermediate/high-structure groups. Each group CDF is blended with the global CDF using a predetermined weight capped at 0.75 and shrunk by contributing image count. No test ground truth enters feature computation or group selection. Post-calibration rescales the full quantile function around the frozen reconstruction; raw and calibrated scores are reported separately because 95% calibration need not improve interior quantiles. Context-dependent marginals do not model joint spatial error correlations. Detailed subgroup metrics and paired image-bootstrap intervals are in comparison.json.']
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
