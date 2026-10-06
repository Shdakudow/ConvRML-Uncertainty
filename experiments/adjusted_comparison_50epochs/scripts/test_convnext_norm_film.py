import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.shamus_rml import ShamusRML
from models.convnext_norm_film import FiLMBackbone,NormFiLM
class Tests(unittest.TestCase):
 def test_backbone_identity_and_gradients(self):
  torch.manual_seed(42);base=ShamusRML().eval();y=torch.rand(2,3,32,32)
  with torch.no_grad():expected=base(y)
  model=FiLMBackbone(base).eval()
  self.assertEqual(len(model.locations),48)
  self.assertEqual(sum(r['channels_last'] for r in model.locations),36)
  with torch.no_grad():torch.testing.assert_close(model(y),expected,rtol=0,atol=0)
  model(y,torch.tensor([.05,.95])).square().mean().backward()
  self.assertTrue(all(f.projection[-1].weight.grad is not None for f in model.films.values()))
  self.assertIsNone(model._emb)
  with self.assertRaises(RuntimeError):
   with model.condition(y):raise RuntimeError('interrupt')
  self.assertIsNone(model._emb)
 def test_layouts(self):
  emb=torch.randn(2,512)
  for last,shape in [(True,(2,4,4,32)),(False,(2,32,4,4))]:
   f=NormFiLM(32,last);x=torch.randn(*shape)
   torch.testing.assert_close(f(x,emb),x)
   with torch.no_grad():f.projection[-1].bias[32:].fill_(.2)
   torch.testing.assert_close(f(x,emb),x+.2)
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
