import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from models.corrected_uncertainty import (PinballQUTCC,EndToEndIm2Im,FullDropoutRML,
    training_loss,experiment_seeds,check_resume,PROTOCOL)
from models.formal_quantile import pinball
from baselines.mc_dropout import dropout_inference
from scripts.asymmetric_rcps import select_asymmetric_factors

class TinyBase(nn.Module):
    def __init__(self):
        super().__init__();self.feature=nn.Conv2d(3,4,1);self.up2=nn.Identity();self.head=nn.Conv2d(4,3,3,padding=1)
    def features(self,y):return self.feature(y)

class Tests(unittest.TestCase):
    def test_independent_shuffle_and_exact_resume(self):
        orders=[]
        for member in range(10):
            s=experiment_seeds(f'ensemble_{member:02d}')
            g=torch.Generator().manual_seed(s['data']);first=torch.randperm(80,generator=g)
            saved=g.get_state();second=torch.randperm(80,generator=g)
            g2=torch.Generator();g2.set_state(saved)
            self.assertTrue(torch.equal(second,torch.randperm(80,generator=g2)))
            self.assertFalse(torch.equal(first,second));orders.append(tuple(first.tolist()))
        self.assertEqual(len(set(orders)),10)
        self.assertEqual(len({experiment_seeds(f'ensemble_{i:02d}')['initialization'] for i in range(10)}),10)
        with self.assertRaises(ValueError):check_resume({},'im2im','split')
        check_resume(dict(protocol=PROTOCOL,arm='im2im',manifest_sha256='split',seeds=experiment_seeds('im2im')),'im2im','split')

    def test_im2im_all_outputs_update_backbone_and_heads(self):
        model=EndToEndIm2Im(TinyBase());y=torch.rand(2,3,8,8);x=torch.rand_like(y)
        center,lo,hi=model.bounds(y)
        losses=[(center-x).square().mean(),pinball(lo,x,torch.full((2,),.05)),pinball(hi,x,torch.full((2,),.95))]
        for loss in losses:
            grad=torch.autograd.grad(loss,model.base.feature.weight,retain_graph=True)[0]
            self.assertGreater(grad.abs().sum().item(),0)
        torch.testing.assert_close(training_loss(model,'im2im',y,x),sum(losses))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        sum(losses).backward()
        for chunk in model.base.head.weight.grad.chunk(3):self.assertGreater(chunk.abs().sum().item(),0)

    def test_qutcc_pinball_only_one_forward_and_median(self):
        model=PinballQUTCC();y=torch.rand(2,3,32,32);x=torch.rand_like(y);q=torch.tensor([.1,.9]);calls=[]
        handle=model.base.register_forward_hook(lambda *args:calls.append(1))
        loss=training_loss(model,'conditioned',y,x,q);self.assertEqual(len(calls),1)
        loss.backward();handle.remove()
        self.assertTrue(all(f.projection[-1].weight.grad is not None for f in model.base.films.values()))
        self.assertFalse(hasattr(model,'head'))  # No extra MSE reconstruction head.
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loss.detach(),pinball(model(y,q),x,q))
            center,_,_=model.bounds(y);torch.testing.assert_close(center,model(y,torch.full((2,),.5)))

    def test_dropout_encoder_decoder_and_inference_mode_restoration(self):
        model=FullDropoutRML(.1);y=torch.rand(2,3,32,32)
        self.assertEqual(len(model.encoder_dropout),36);self.assertEqual(len(model.dropout_locations),48)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(y),model(y),rtol=0,atol=0)
            with dropout_inference(model):
                self.assertTrue(all(d.training for d in model.encoder_dropout.values()))
                self.assertFalse(torch.equal(model(y),model(y)))
            self.assertTrue(all(not m.training for m in model.modules()))
        model.train();training_loss(model,'mc_010',y,torch.rand_like(y)).backward()
        self.assertIsNotNone(next(model.encoder.parameters()).grad)

    def test_asymmetric_scales_and_joint_confidence_budget(self):
        rng=np.random.default_rng(2);residual=rng.normal(-1,.8,size=(512,3,4,4))
        lower=-residual;upper=residual
        factors,audit,losses=select_asymmetric_factors(lower,upper)
        self.assertGreater(factors['rcps_hb']['lower'],factors['rcps_hb']['upper'])
        self.assertEqual(audit['tail_alpha'],.05);self.assertEqual(audit['tail_delta'],.025)
        self.assertTrue(all(p<.025 for p in audit['tail_hb_pvalues'].values()))
        self.assertLessEqual(losses['rcps_hb'].mean(),.1)
        for side in ('lower','upper'):self.assertGreaterEqual(factors['rcps_hb'][side],factors['empirical'][side])
        with self.assertRaises(ValueError):select_asymmetric_factors(lower,upper[:2])

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
