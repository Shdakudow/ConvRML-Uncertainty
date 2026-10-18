import unittest,copy
import torch
from torch import nn
from model import pinball,learning_rate,training_loss
from calibration import search_quantiles,tail_budget

class Toy(nn.Module):
    method='qutcc'
    def __init__(self):super().__init__();self.bias=nn.Parameter(torch.tensor(.2));self.seen=None
    def forward(self,y,q):self.seen=q.detach().clone();return y+self.bias+q[:,None,None,None]*.1

class Tests(unittest.TestCase):
    def test_pinball_direction_and_rgb(self):
        x=torch.tensor([1.,0.]).reshape(2,1,1,1).expand(2,3,2,2);pred=torch.full_like(x,.25,requires_grad=True);q=torch.tensor([.9,.1])
        loss=pinball(pred,x,q);self.assertAlmostEqual(float(loss),.45,places=6);loss.backward()
        self.assertTrue((pred.grad[0]<0).all());self.assertTrue((pred.grad[1]>0).all())
    def test_only_sampled_pinball_updates(self):
        m=Toy();y=torch.zeros(10000,3,1,1);x=torch.ones_like(y);g=torch.Generator().manual_seed(1)
        loss=training_loss(m,y,x,g);loss.backward();q=m.seen
        self.assertFalse(q.requires_grad);self.assertLess(abs(float(q.mean())-.5),.01)
        self.assertAlmostEqual(float(m.bias.grad),-float(q.mean()),places=6)
        self.assertTrue((q<.05).any() and (q>.95).any())
    def test_schedule_continuity_on_resume(self):
        steps=15000;total=50*steps
        self.assertAlmostEqual(learning_rate(0,total),1e-5)
        self.assertAlmostEqual(learning_rate(15*steps,total),1e-4)
        self.assertAlmostEqual(learning_rate(total,total),1e-9)
        saved_step=2*steps;self.assertNotEqual(learning_rate(saved_step,total),learning_rate(0,total))
    def test_optimizer_rng_resume(self):
        m=Toy();opt=torch.optim.Adam(m.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=0)
        rng=torch.Generator().manual_seed(12);y=torch.zeros(4,3,1,1);x=torch.ones_like(y)
        def step(model,optimizer,g,number):
            optimizer.zero_grad();optimizer.param_groups[0]['lr']=learning_rate(number,100)
            loss=training_loss(model,y,x,g);loss.backward();optimizer.step()
        step(m,opt,rng,0);state=copy.deepcopy((m.state_dict(),opt.state_dict(),rng.get_state()))
        step(m,opt,rng,1)
        n=Toy();n.load_state_dict(state[0]);o=torch.optim.Adam(n.parameters());o.load_state_dict(state[1]);g=torch.Generator();g.set_state(state[2]);step(n,o,g,1)
        torch.testing.assert_close(m.bias,n.bias,rtol=0,atol=0);torch.testing.assert_close(m.seen,n.seen,rtol=0,atol=0)
    def test_calibration_known_uniform_distribution(self):
        x=torch.linspace(0,1,1000)
        def risk(lo,hi):return float((x<lo).float().mean()),float((x>hi).float().mean())
        r=search_quantiles(risk,1000);self.assertLessEqual(risk(r['lower_q'],r['upper_q'])[0],tail_budget(.1,1000))
        self.assertLess(abs(r['lower_q']-.05),.003);self.assertLess(abs(r['upper_q']-.95),.003)
    def test_calibration_refuses_uncovered_extremes(self):
        with self.assertRaises(RuntimeError):search_quantiles(lambda l,u:(.2,.2),1000)
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
