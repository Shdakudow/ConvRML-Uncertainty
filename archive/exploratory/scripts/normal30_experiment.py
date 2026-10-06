"""Prepare isolated exact-epoch-20 continuation and compare held-out reconstructions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from models.shamus_rml import ShamusRML
from models.formal_quantile import QuantileRML
from scripts.formal_rml_train import Pairs,DATA

SOURCE=Path('results/formal_quantile_comparison')
OUT=Path('results/normal_convrml_epoch30_20260924')

def prepare():
    if (OUT/'prepared.json').exists():
        print('Already prepared; keeping continuation state.');return
    manifest=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
    ck=torch.load(SOURCE/'normal/latest.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==20 and ck['manifest_sha256']==manifest
    assert len(ck['history'])==20
    record={'source':str(SOURCE/'normal/latest.pt'),'start_epoch':20,'target_epoch':30,
            'manifest_sha256':manifest,'optimizer_resumed':True,'rng_resumed':True,
            'optimizer_lrs':[g['lr'] for g in ck['optimizer']['param_groups']],
            'training_objective':'MSE only','train_count':20000,'validation_count':2500,
            'test_count':1000,'chunks':[22,24,26,28,30]}
    del ck
    (OUT/'normal').mkdir(parents=True,exist_ok=True)
    for name in ['latest.pt','best.pt','history.json']:
        shutil.copy2(SOURCE/'normal'/name,OUT/'normal'/name)
    shutil.copy2(SOURCE/'smoke_normal.json',OUT/'smoke_normal.json')
    (OUT/'prepared.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))

@torch.inference_mode()
def evaluate(kind='normal'):
    torch.set_num_threads(2)
    info=json.loads((OUT/'prepared.json').read_text())
    manifest=json.loads((DATA/'manifest.json').read_text())
    paths={'original_epoch20':SOURCE/'normal/latest.pt',
           'continued_epoch30':OUT/kind/'latest.pt',
           'best_validation_through30':OUT/kind/'best.pt',
           'original_best_validation':SOURCE/'normal/best.pt'}
    loader=DataLoader(Subset(Pairs('test'),range(1000)),batch_size=2,num_workers=2)
    report={};examples={};allrows=[]
    for label,path in paths.items():
        base=ShamusRML()
        conditioned=kind=='conditioned' and label in ['continued_epoch30','best_validation_through30']
        model=(QuantileRML(base,'conditioned') if conditioned else base).cuda().eval()
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['manifest_sha256']==info['manifest_sha256']
        if label=='original_epoch20':assert ck['epoch']==20
        if label=='continued_epoch30':assert ck['epoch']==30
        model.load_state_dict(ck['model']);epoch=ck['epoch'];del ck
        rows=[];position=0;sample=[]
        for b in loader:
            y,x=b[:,0].cuda(),b[:,1].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):r=(model.features(y)[1] if conditioned else model(y)).float()
            mse=(r.double()-x.double()).square().mean((1,2,3))
            psnr=-10*torch.log10(mse.clamp_min(1e-30))
            similarity=ssim(r,x,data_range=1.,reduction='none')
            for i in range(len(y)):
                row={'model':label,'image_id':manifest['test'][position+i]['id'],
                     'psnr_db':psnr[i].item(),'ssim':similarity[i].item(),'mse':mse[i].item()}
                rows.append(row);allrows.append(row)
            if position<8:sample.append(r.cpu().numpy())
            position+=len(y)
        examples[label]=np.concatenate(sample)
        report[label]={'epoch':epoch,'count':len(rows),
                       **{key:float(np.mean([r[key] for r in rows])) for key in ['psnr_db','ssim','mse']}}
        del model,base
    import csv
    with (OUT/'per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    pairs=np.load(DATA/'pairs.npy',mmap_mode='r')
    examples.update(measurement=pairs[22500:22508,0].astype(np.float32)/255,
                    target=pairs[22500:22508,1].astype(np.float32)/255,
                    ids=np.array([r['id'] for r in manifest['test'][:8]]))
    np.savez_compressed(OUT/'examples.npz',**examples)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2))
    lines=['# Normal ConvRML: 20 versus 30 epochs','',
           'Exact epoch-20 continuation with model, optimizer and RNG resumed; five two-epoch jobs. Same 20k training and 2500 validation pairs. Evaluation uses the same first 1000 test pairs; best checkpoint selection uses validation MSE only.', '',
           '| Checkpoint | Epoch | Mean per-image PSNR | Mean SSIM |', '|---|---:|---:|---:|']
    for label,r in report.items():lines.append(f"| {label} | {r['epoch']} | {r['psnr_db']:.4f} | {r['ssim']:.5f} |")
    lines+=['',f'Run directory: {OUT}; model kind: {kind}. This report evaluates reconstruction only; interval coverage and width are not evaluated. See prepared.json for training settings. Earlier fixed/Q heads were trained on other backbone features and must not be assumed valid on this updated model. Original best-versus-new best and exact epoch20-versus-epoch30 are reported separately. Single training trajectory; no statistical significance claim.']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate']);p.add_argument('--root',type=Path,default=OUT);p.add_argument('--kind',choices=['normal','conditioned'],default='normal');a=p.parse_args()
    OUT=a.root
    prepare() if a.action=='prepare' else evaluate(a.kind)
