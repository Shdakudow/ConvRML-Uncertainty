"""Paper-style objectives with the explicitly requested ConvNeXt/FPN substitution."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.shamus_rml import ShamusRML
from models.convnext_norm_film import FiLMBackbone

def pinball(pred,target,q):
    e=target-pred;q=q.reshape(-1,1,1,1)
    return torch.maximum(q*e,(q-1)*e).mean()

def learning_rate(step,total_steps,peak=1e-4):
    """Reference train.py: 30% linear warmup, then cosine; never resets per job."""
    pct=step/total_steps;low=peak/10
    return low+(pct/.3)*(peak-low) if pct<=.3 else low/1e4+(peak-low/1e4)/2*(math.cos(math.pi*(pct-.3)/.7)+1)

class Model(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        base=ShamusRML()
        if method=='qutcc':self.net=FiLMBackbone(base)
        elif method=='im2im':
            old=base.head;base.head=nn.Conv2d(old.in_channels,9,old.kernel_size,padding=old.padding)
            self.net=base
        else:raise ValueError(method)
    def forward(self,y,q=None):
        h,w=y.shape[-2:];padded=F.pad(y,(0,(-w)%32,0,(-h)%32),mode='replicate')
        if self.method=='qutcc':
            if q is None:q=y.new_full((len(y),),.5)
            return self.net(padded,q)[...,:h,:w]
        features=self.net.up2(self.net.features(padded))
        return (self.net.head(features).reshape(len(y),3,3,*padded.shape[-2:])+padded[:,None])[...,:h,:w]
    def center(self,y):return self(y) if self.method=='qutcc' else self(y)[:,1]

def training_loss(model,y,x,generator):
    if model.method=='qutcc':
        q=torch.rand(len(y),device=y.device,dtype=torch.float32,generator=generator)
        return pinball(model(y,q),x,q)
    out=model(y)
    return (out[:,1]-x).square().mean()+pinball(out[:,0],x,x.new_full((len(x),),.05))+pinball(out[:,2],x,x.new_full((len(x),),.95))
