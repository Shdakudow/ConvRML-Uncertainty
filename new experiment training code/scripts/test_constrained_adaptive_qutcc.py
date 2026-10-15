import sys,copy,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from models.constrained_adaptive_qutcc import CoverageController,ConstrainedAdaptiveQUTCC,interval_terms,joint_step

class Toy(nn.Module):
 def __init__(self):super().__init__();self.slope=nn.Parameter(torch.tensor(.7))
 def forward(self,y,q=None):
  if q is None:q=y.new_full((len(y),),.5)
  return .5+self.slope*(q[:,None,None,None]-.5)+y*.02

class Tests(unittest.TestCase):
 def test_constraint_feedback(self):
  c=CoverageController(momentum=0);initial=float(c.multiplier)
  c.update(torch.tensor(.4));self.assertGreater(float(c.multiplier),initial)
  high=float(c.multiplier);c.update(torch.tensor(.01));self.assertLess(float(c.multiplier),high)
  c.multiplier.zero_();c.update(torch.tensor(0.));self.assertEqual(float(c.multiplier),0)
 def test_coverage_boundaries_and_direction(self):
  x=torch.tensor([0.,.5,1.]);r=interval_terms(torch.zeros(3),torch.ones(3),x)
  self.assertEqual(float(r['hard_risk']),0);self.assertEqual(float(r['soft_risk']),0)
  lo=torch.tensor([.2],requires_grad=True);hi=torch.tensor([.8],requires_grad=True)
  t=interval_terms(lo,hi,torch.tensor([.1]));t['soft_risk'].backward()
  self.assertGreater(float(lo.grad),0)
  self.assertLess(float(interval_terms(torch.tensor([.15]),hi.detach(),torch.tensor([.1]))['soft_risk']),float(t['soft_risk']))
 def test_joint_interval_gradients_and_resume(self):
  torch.manual_seed(42);m=ConstrainedAdaptiveQUTCC(Toy());y=torch.rand(3,3,16,16);x=torch.full_like(y,.5)
  r=m.interval_prediction(y);terms=interval_terms(r['lower'],r['upper'],x)
  (terms['width']+terms['soft_risk']).backward()
  self.assertGreater(float(m.quantile_model.slope.grad.abs()),0)
  self.assertGreater(float(m.selector.head.bias.grad.abs().sum()),0)
  opt=torch.optim.AdamW(m.parameters(),lr=1e-4);g=torch.Generator().manual_seed(11)
  before=m.selector.head.bias.detach().clone();joint_step(m,y,x,opt,g,amp=False)
  self.assertFalse(torch.equal(before,m.selector.head.bias))
  state=copy.deepcopy((m.state_dict(),opt.state_dict(),g.get_state()))
  expected=joint_step(m,y,x,opt,g,amp=False)
  n=ConstrainedAdaptiveQUTCC(Toy());n.load_state_dict(state[0]);o=torch.optim.AdamW(n.parameters(),lr=1e-4);o.load_state_dict(state[1]);h=torch.Generator();h.set_state(state[2])
  actual=joint_step(n,y,x,o,h,amp=False);self.assertEqual(expected,actual)
  for a,b in zip(m.state_dict().values(),n.state_dict().values()):torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
