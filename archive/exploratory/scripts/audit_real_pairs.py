import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import tifffile
from PIL import Image
from omegaconf import OmegaConf
from dataset import get_loader
out=Path('results/real_pair_audit');out.mkdir(parents=True,exist_ok=True)
c=OmegaConf.load('configs/PLD_RML/config_real_small.yaml')
loaders=get_loader(c);report={};flags=[]
for split,loader in zip(['train','val','test'],loaders):
 ds=loader.dataset;raw=ds.dataset;indices=ds.indices;prev=[];mismatches=[]
 for i in indices:
  mn,tn=raw.data_list[i],raw.target_list[i]
  mid=int(re.search(r'img_(\d+)',mn).group(1));tid=int(re.search(r'img_(\d+)',tn).group(1))
  if mid!=tid:mismatches.append([mn,tn])
  def thumb(path):
   a=tifffile.imread(path)[...,:3];return np.asarray(Image.fromarray(a).resize((48,30)),dtype=float)/255
  y=thumb(Path(raw.data_dir)/mn);x=thumb(Path(raw.target_dir)/tn)
  for pid,px,py in prev[-3:]:
   td=float(np.mean(np.abs(x-px)));md=float(np.mean(np.abs(y-py)))
   if td<.003 and md>.012:flags.append(dict(split=split,ids=[pid,mid],target_mae=td,measurement_mae=md))
  prev.append((mid,x,y))
 report[split]={'count':len(indices),'filename_mismatches':mismatches}
report['near_duplicate_target_flags']=flags
report['limitations']='Heuristic screen for repeated target frames with changing measurements; cannot certify semantic correspondence or detect all mismatches. No automatic training exclusions.'
(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
assert all(not report[s]['filename_mismatches'] for s in ['train','val','test'])
