import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.qutcc_original.adapter import OriginalQUTCC,OriginalIm2Im
from models.qutcc_original.quantile_uqnet import ResBlock,AttentionBlock
class Tests(unittest.TestCase):
 def test_sites_and_rgb(self):
  with torch.device('meta'):
   q=OriginalQUTCC();i=OriginalIm2Im()
  self.assertEqual([sum(isinstance(m,ResBlock) and m.use_scale_shift_norm for m in getattr(q.base,k).modules()) for k in ['input_blocks','middle_block','output_blocks']],[11,2,17])
  self.assertEqual(q.base.input_blocks[0][0].in_channels,3)
  self.assertEqual(q.base.out[-1].out_channels,3)
  self.assertEqual(i.prediction.out_channels,3)
  self.assertFalse(any(isinstance(m,AttentionBlock) for m in i.modules()))
 def test_checkpoint_gradients_and_inference(self):
  net=AttentionBlock(64);x=torch.randn(2,64,4,4,requires_grad=True)
  with torch.autocast('cpu',dtype=torch.bfloat16):out=net(x)
  out.float().square().mean().backward()
  self.assertTrue(torch.isfinite(x.grad).all())
  with torch.inference_mode():self.assertEqual(net(x.detach()).shape,x.shape)
 def test_quantile_gradient(self):
  block=ResBlock(32,512,.0,use_scale_shift_norm=True)
  with torch.no_grad():block.out_layers[-1].weight.normal_(0,.01)
  y=block(torch.randn(2,32,4,4),torch.randn(2,512));y.square().mean().backward()
  self.assertGreater(float(block.emb_layers[-1].weight.grad.abs().sum()),0)
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
