"""Export aligned, calibrated examples from existing predictions; no inference."""
import json,hashlib
from pathlib import Path
import numpy as np

N=8
data=Path('data/formal_rml_25k')
qroot=Path('results/formal_quantile_comparison')
broot=Path('results/formal_rml_25k_epoch20')
calroot=Path('results/formal_rml_calibration95')
out=Path('results/four_model_examples');out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((data/'manifest.json').read_text())
manifest_hash=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
qreport=json.loads((qroot/'comparison.json').read_text())
breport=json.loads((broot/'comparison.json').read_text())
bfactors=json.loads((calroot/'factors.json').read_text())
assert breport['manifest_sha256']==manifest_hash
assert all(v['train_manifest_sha256']==manifest_hash for v in qreport['selected_checkpoints'].values())
pairs=np.load(data/'pairs.npy',mmap_mode='r')
payload={'measurement':pairs[22500:22500+N,0].astype(np.float32)/255,'target':pairs[22500:22500+N,1].astype(np.float32)/255,'ids':np.array([v['id'] for v in manifest['test'][:N]])}
metadata={'selection':'First eight entries of the fixed test manifest; no visual selection','manifest_sha256':manifest_hash,'coverage_target':.95,'methods':{}}
for kind,label in [('fixed','im2im-style fixed head'),('conditioned','Q-conditioned ConvRML')]:
    a={k:np.array(np.load(qroot/kind/f'test_{k}.npy',mmap_mode='r')[:N]) for k in ('r','lo','hi')}
    factor=qreport['factors'][kind];wl=np.maximum(a['r']-a['lo'],.001);wu=np.maximum(a['hi']-a['r'],.001)
    payload[kind+'_mean']=a['r'];payload[kind+'_lo']=a['r']-factor*wl;payload[kind+'_hi']=a['r']+factor*wu
    metadata['methods'][kind]={'label':label,'factor':factor,'selected_epoch':qreport['selected_checkpoints'][kind]['selected_epoch']}
for kind,workers in [('ensemble',range(5)),('mc',[5])]:
    means=np.stack([np.array(np.load(broot/f'worker_{i}/mean.npy',mmap_mode='r')[:N]) for i in workers])
    variances=np.stack([np.array(np.load(broot/f'worker_{i}/variance.npy',mmap_mode='r')[:N]) for i in workers])
    mean=means.mean(0);std=np.maximum(np.sqrt(variances.mean(0)+means.var(0)),bfactors['std_floor'])
    radius=bfactors['radius_factors'][kind]*std
    payload[kind+'_mean']=mean;payload[kind+'_lo']=mean-radius;payload[kind+'_hi']=mean+radius
    metadata['methods'][kind]={'label':'Deep ensemble (5 members)' if kind=='ensemble' else 'MC dropout (30 samples)','factor':bfactors['radius_factors'][kind]}
for v in payload.values():assert np.isfinite(v).all()
np.savez_compressed(out/'examples.npz',**payload)
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps({'count':N,'ids':payload['ids'].tolist(),'output':str(out)},indent=2))
