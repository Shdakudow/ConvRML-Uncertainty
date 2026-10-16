"""Check exact initial weights, shuffle streams, and deterministic control output."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scripts.train_dropout_control import make_model,experiment_seeds,ROOT
from scripts.corrected_comparison_study import make_model as make_reference

def main():
 torch.set_num_threads(2)
 a=make_model('mc_000');b=make_reference('mc_005')
 assert a.state_dict().keys()==b.state_dict().keys()
 h=hashlib.sha256()
 for k,v in a.state_dict().items():
  assert torch.equal(v,b.state_dict()[k]),k
  h.update(k.encode());h.update(v.numpy().tobytes())
 assert len(a.dropout_locations)==48
 assert all(m.p==0 for m in a.modules() if isinstance(m,nn.Dropout))
 assert sum(m.p==.05 for m in b.modules() if isinstance(m,nn.Dropout))==48
 assert all(m.p in (0.,.05) for m in b.modules() if isinstance(m,nn.Dropout))
 y=torch.rand(1,3,64,64)
 with torch.no_grad():
  a.eval();b.eval();torch.testing.assert_close(a(y),b(y),rtol=0,atol=0)
  a.train();torch.testing.assert_close(a(y),a(y),rtol=0,atol=0)
  b.train();assert not torch.equal(b(y),b(y))
 seed=experiment_seeds('mc_000')['data'];g1=torch.Generator().manual_seed(seed);g2=torch.Generator().manual_seed(seed)
 ds=TensorDataset(torch.arange(60000))
 for _ in range(2):
  x=torch.cat([t[0] for t in DataLoader(ds,batch_size=2,shuffle=True,generator=g1)])
  z=torch.cat([t[0] for t in DataLoader(ds,batch_size=2,shuffle=True,generator=g2)])
  assert torch.equal(x,z)
 report=dict(passed=True,initial_state_sha256=h.hexdigest(),identical_initial_weights=True,matched_two_epoch_shuffle=True,zero_dropout_deterministic=True,p005_training_stochastic=True,sites=48)
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'matching_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
