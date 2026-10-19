"""Immutable existing splits. Training never instantiates calibration/test datasets."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
PROJECT=Path(os.environ.get('CONVRML_PROJECT',str(Path.home()/'ConvRML')))
NORMAL={'rml':('optics_intervals_60k_5k_5k_20260928','formal_rml_calibration95'),'diffuser':('diffuser_intervals_60k_5k_5k_20260928','diffuser_intervals_calibration_20260928')}
RAW=PROJECT/'experiments/dlmd_raw_adaptive_size_20261016/data/split.json'

def manifests(camera):
    if camera=='raw':return [RAW]
    return [PROJECT/'data'/n/'manifest.json' for n in NORMAL[camera]]

def identity(camera):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests(camera)}

class Pairs(Dataset):
    def __init__(self,camera,role):
        assert role in ('train','val','test','calibration');self.camera=camera
        if camera=='raw':self.rows=json.loads(RAW.read_text())['roles'][role];self.array=None
        else:
            p=manifests(camera)[role=='calibration'];m=json.loads(p.read_text());self.rows=m[role]
            self.start=0 if role in ('train','calibration') else len(m['train'])+(len(m['val']) if role=='test' else 0)
            self.array=np.load(p.parent/'pairs.npy',mmap_mode='r')
            assert len(self.array)==sum(len(m[k]) for k in (['calibration'] if role=='calibration' else ['train','val','test']))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        if self.array is not None:return torch.from_numpy(np.array(self.array[self.start+i],copy=True)).float()/255
        out=[]
        for key in ('measurement','target'):
            with Image.open(self.rows[i][key]['path']) as im:a=np.asarray(im.convert('RGB')).copy()
            out.append(torch.from_numpy(a).permute(2,0,1).float()/255)
        assert out[0].shape==out[1].shape
        return torch.stack(out)

def audit(camera):
    if camera=='raw':roles=json.loads(RAW.read_text())['roles'];expected={'train':17999,'val':2000,'test':2000,'calibration':3000}
    else:
        roles={};[roles.update({k:v for k,v in json.loads(p.read_text()).items() if k in ('train','val','test','calibration')}) for p in manifests(camera)]
        expected={'train':60000,'val':5000,'test':5000,'calibration':2048}
    assert {k:len(roles[k]) for k in expected}==expected
    seen=set()
    for role,rows in roles.items():
        ids={str(r['id']) for r in rows};assert len(ids)==len(rows) and not seen.intersection(ids),(camera,role,'split overlap');seen.update(ids)
    return expected
