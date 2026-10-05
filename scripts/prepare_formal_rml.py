"""Prepare a fixed, exact-target-deduplicated 25k-pair real-RML split on CPU."""
import sys,json,re,hashlib,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

OUT=Path('data/formal_rml_25k');ROOT=Path('/share/monakhova/shamus_data/parallel_100k')

def image(path):
    with Image.open(path) as im:a=np.array(im.convert('RGB'))
    x=torch.from_numpy(a).permute(2,0,1).float()/255
    h,w=x.shape[-2:];s=256/min(h,w)
    size=[max(256,round(h*s)),max(256,round(w*s))]
    x=F.interpolate(x[None],size=size,mode='bilinear',align_corners=False,antialias=True)[0]
    y,z=[round((v-256)/2) for v in size]
    return (x[:,y:y+256,z:z+256]*255).round().clamp(0,255).byte().numpy()

def main():
    torch.set_num_threads(2)
    if (OUT/'manifest.json').exists():raise FileExistsError('Prepared manifest already exists')
    OUT.mkdir(parents=True,exist_ok=True)
    ids=[]
    for p in (ROOT/'rml').glob('4x_img_*_cam_1.tiff'):
        i=int(re.fullmatch(r'4x_img_(\d+)_cam_1.tiff',p.name)[1])
        if i not in (5,6) and (ROOT/'gt2rml'/f'warped_4x_undistorted_img_{i}_cam_2.tiff').exists():ids.append(i)
    ids.sort();random.Random(20260923).shuffle(ids)
    cache=np.lib.format.open_memmap(OUT/'pairs.npy',mode='w+',dtype=np.uint8,shape=(25000,2,3,256,256))
    records=[];seen=set();rejected=[]
    for i in ids:
        yp=ROOT/'rml'/f'4x_img_{i}_cam_1.tiff';xp=ROOT/'gt2rml'/f'warped_4x_undistorted_img_{i}_cam_2.tiff'
        try:
            x=image(xp);key=hashlib.sha256(x.tobytes()).hexdigest()
            if key in seen:rejected.append({'id':i,'reason':'duplicate processed target'});continue
            y=image(yp)
        except Exception as e:rejected.append({'id':i,'reason':str(e)});continue
        n=len(records);cache[n,0]=y;cache[n,1]=x;seen.add(key)
        records.append({'id':i,'measurement':str(yp),'target':str(xp),'target_sha256':key})
        if len(records)%1000==0:cache.flush();print('PREPARED',len(records),flush=True)
        if len(records)==25000:break
    if len(records)!=25000:raise RuntimeError(f'Only {len(records)} unique usable pairs')
    cache.flush()
    report={'seed':20260923,'train':records[:20000],'val':records[20000:22500],'test':records[22500:],'rejected':rejected,'notes':['Exact processed target deduplication; near duplicates/acquisition groups not certified.','Paired by camera filename ID; semantic pairing not fully audited.','RGB 256 resize-shortside/center-crop, cached uint8.','No pretrained checkpoint loaded by training.']}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print('READY: 20000 train, 2500 val, 2500 test',flush=True)

if __name__=='__main__':main()
