import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from models.convnext import FNN
from baselines.convrml_heads import mc_head,gaussian_head
from baselines.mc_dropout import MCDropout
from baselines.deep_ensemble import DeepEnsembleMember
from baselines.common import aggregate_moments

class Tests(unittest.TestCase):
    def test_dropout_and_modes(self):
        m=nn.Sequential(nn.Conv2d(3,3,1),nn.BatchNorm2d(3),nn.Dropout2d(.4))
        m.train();original=m[1].running_mean.clone()
        wrapper=MCDropout(m,num_mc_samples=8)
        x=torch.randn(4,3,8,8);samples=wrapper.predict_samples(x)
        self.assertGreater(samples.var(0).sum().item(),0)
        torch.testing.assert_close(m[1].running_mean,original)
        self.assertTrue(all(layer.training for layer in m.modules()))
        self.assertEqual(wrapper(x).shape,(4,9,8,8))

    def test_moments(self):
        means=torch.tensor([0.,2.]).reshape(2,1,1,1,1);variances=torch.ones_like(means)*3
        mean,total,aleatoric,epistemic=aggregate_moments(means,variances)
        self.assertEqual(mean.item(),1);self.assertEqual(total.item(),4)
        self.assertEqual(aleatoric.item(),3);self.assertEqual(epistemic.item(),1)

    def test_rgb_heads_learn(self):
        template=FNN(16,3,(16,16,8,8));f=torch.randn(2,16,8,8)
        mc=mc_head(template)
        self.assertFalse(torch.equal(mc(f),mc(f)))
        m=DeepEnsembleMember(gaussian_head(template,seed=1))
        mean,var=m(f);self.assertEqual(mean.shape,(2,3,8,8));self.assertTrue((var>0).all())
        m.loss(f,torch.rand_like(mean)).backward()
        self.assertGreater(m.baseModel.f[-1].weight.grad.abs().sum().item(),0)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
