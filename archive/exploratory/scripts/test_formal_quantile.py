import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from models.formal_quantile import QuantileRML,pinball

class SmallBase(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Conv2d(3,256,1);self.up2=nn.Identity();self.head=nn.Conv2d(256,3,3,padding=1)

class Tests(unittest.TestCase):
    def test_native_ninety_percent_levels(self):
        m=QuantileRML(SmallBase(),'conditioned',levels=(.05,.95))
        seen=[]
        original=m.predict
        def record(y,f,q=None):
            seen.append(float(q[0]));return original(y,f,q)
        m.predict=record
        m.bounds(torch.rand(1,3,8,8))
        self.assertAlmostEqual(seen[0],.05,places=6)
        self.assertAlmostEqual(seen[1],.95,places=6)

    def test_pinball_gradient_direction(self):
        # Both endpoints must move toward a target outside the interval.
        for q in (.05,.95):
            p=torch.zeros(1,1,1,1,requires_grad=True)
            pinball(p,torch.ones_like(p),torch.tensor([q])).backward()
            self.assertAlmostEqual(p.grad.item(),-q,places=6)

    def test_fixed_freezes_and_matches_reconstruction(self):
        m=QuantileRML(SmallBase(),'fixed').train();y=torch.rand(2,3,8,8)
        r,l,h=m.bounds(y);torch.testing.assert_close(r,l);torch.testing.assert_close(r,h)
        (l-y).square().mean().backward()
        self.assertFalse(m.base.training)
        self.assertTrue(all(p.grad is None for p in m.base.parameters()))
        self.assertGreater(m.head.weight.grad.abs().sum().item(),0)

    def test_conditioned_trains_backbone_and_embedding(self):
        m=QuantileRML(SmallBase(),'conditioned');y=torch.rand(2,3,8,8);x=torch.rand_like(y)
        f,r=m.features(y);q=torch.tensor([.1,.9]);pred=m.predict(y,f,q)
        torch.testing.assert_close(pred,r)
        (pinball(pred,x,q)+(r-x).square().mean()).backward()
        self.assertGreater(m.base.features.weight.grad.abs().sum().item(),0)
        self.assertGreater(m.conditioner.film.weight.grad.abs().sum().item(),0)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
