"""Prepare DiffuserCam data with precisely the same scene split as RML."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.prepare_formal_rml import image, ROOT

OUT=Path('data/diffuser_intervals_60k_5k_5k_20260928')
CAL=Path('data/diffuser_intervals_calibration_20260928')
SOURCE=Path('data/optics_intervals_60k_5k_5k_20260928/manifest.json')
CALSOURCE=Path('data/formal_rml_calibration95/manifest.json')


def source_paths(ident):
    return (ROOT/'diffuser'/f'4x_img_{ident}_cam_0.tiff',
            ROOT/'gt2dc'/f'warped_4x_undistorted_img_{ident}_cam_2.tiff')


def main():
    torch.set_num_threads(2)
    if (OUT/'manifest.json').exists() or (CAL/'manifest.json').exists():
        raise FileExistsError('Prepared DiffuserCam metadata exists; inspect before retry')
    source=json.loads(SOURCE.read_text())
    roles={role:source[role] for role in ('train','val','test')}
    roles['calibration']=json.loads(CALSOURCE.read_text())['calibration']
    expected={'train':60000,'val':5000,'test':5000,'calibration':2048}
    if any(len(roles[role])!=count for role,count in expected.items()):
        raise ValueError('Unexpected source split counts')
    ids=[r['id'] for rows in roles.values() for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Scene ID overlaps between source splits')
    missing=[str(p) for ident in ids for p in source_paths(ident) if not p.is_file()]
    if missing:raise FileNotFoundError(f'{len(missing)} missing paired files: {missing[:8]}')
    OUT.mkdir(parents=True,exist_ok=True);CAL.mkdir(parents=True,exist_ok=True)
    cache=np.lib.format.open_memmap(OUT/'pairs.npy',mode='w+',dtype=np.uint8,shape=(70000,2,3,256,256))
    calibration=np.lib.format.open_memmap(CAL/'pairs.npy',mode='w+',dtype=np.uint8,shape=(2048,2,3,256,256))
    hashes={};records={role:[] for role in roles};within_role_duplicates=[]
    position=0
    for role,rows in roles.items():
        for i,row in enumerate(rows):
            yp,xp=source_paths(row['id'])
            y,x=image(yp),image(xp)
            digest=hashlib.sha256(x.tobytes()).hexdigest()
            if digest in hashes:
                previous=hashes[digest]
                if previous['role']!=role:
                    failure={'role':role,'id':row['id'],'previous':previous,'reason':'cross-split duplicate processed DiffuserCam target'}
                    (OUT/'audit_failure.json').write_text(json.dumps(failure,indent=2))
                    raise ValueError(f'Cross-split duplicate target: {failure}')
                within_role_duplicates.append({'role':role,'id':row['id'],'same_as':previous['id']})
            else:hashes[digest]={'role':role,'id':row['id']}
            target_cache=calibration if role=='calibration' else cache
            index=i if role=='calibration' else position
            target_cache[index,0]=y;target_cache[index,1]=x
            if role!='calibration':position+=1
            records[role].append({'id':row['id'],'measurement':str(yp),'target':str(xp),'target_sha256':digest})
            if (i+1)%1000==0:
                target_cache.flush();print(role,i+1,'/',len(rows),flush=True)
    cache.flush();calibration.flush()
    common={'camera':'DiffuserCam','source_rml_manifest_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'source_rml_calibration_manifest_sha256':hashlib.sha256(CALSOURCE.read_bytes()).hexdigest(),
            'within_role_duplicates':within_role_duplicates,
            'notes':['Identical scene IDs/order/split assignments to RML.',
                     'DiffuserCam measurements paired with gt2dc, not gt2rml.',
                     'No exact processed target overlap across roles; within-role duplicates retained and reported to preserve paired-camera comparison.',
                     'RGB uint8 resize-shortside/center-crop to 256; divide by 255 in training.',
                     'Different camera warps/crops mean cross-camera PSNR is not a pure noise-only comparison.',
                     'Near duplicates and acquisition groups are not certified.']}
    for directory,manifest in [(CAL,{**common,'calibration':records['calibration']}),
                               (OUT,{**common,**{r:records[r] for r in ('train','val','test')}})]:
        temporary=directory/'manifest.tmp';temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(directory/'manifest.json')
    print('READY: DiffuserCam 60000/5000/5000 + 2048 calibration',flush=True)


if __name__=='__main__':main()
