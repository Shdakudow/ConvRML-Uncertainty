"""Real-RML normal reconstruction plus fixed or general final-layer quantiles."""
import copy
import torch
from torch import nn
from models.shamus_rml import ShamusRML
from models.convnext import QuantileConditioner

LEVELS=(.025,.975)
def pinball(p,x,q):
    e=x-p;q=q.reshape(-1,1,1,1)
    return torch.maximum(q*e,(q-1)*e).mean()

class QuantileRML(nn.Module):
    def __init__(self,base,kind,levels=LEVELS):
        super().__init__();self.base=base;self.kind=kind
        self.levels=tuple(float(q) for q in levels)
        if len(self.levels)!=2 or not 0<self.levels[0]<self.levels[1]<1:raise ValueError('Invalid endpoint levels')
        if kind=='fixed':
            self.base.requires_grad_(False).eval()
            self.head=nn.Conv2d(256,6,3,padding=1)
            with torch.no_grad():
                self.head.weight.copy_(base.head.weight.repeat(2,1,1,1))
                self.head.bias.copy_(base.head.bias.repeat(2))
        elif kind=='conditioned':
            self.conditioner=QuantileConditioner(256,embedding_dim=64,condition_channels=16)
            self.head=copy.deepcopy(base.head)
        else:raise ValueError(kind)

    def train(self,mode=True):
        super().train(mode)
        if self.kind=='fixed':self.base.eval()
        return self

    def features(self,y):
        if self.kind=='fixed':
            with torch.no_grad():f=self.base.up2(self.base.features(y));r=y+self.base.head(f)
        else:f=self.base.up2(self.base.features(y));r=y+self.base.head(f)
        return f,r

    def predict(self,y,f,q=None):
        if self.kind=='fixed':return self.head(f).reshape(len(y),2,3,*y.shape[-2:])+y[:,None]
        return y+self.head(self.conditioner(f,q))

    def bounds(self,y):
        f,r=self.features(y)
        if self.kind=='fixed':p=self.predict(y,f);return r,p[:,0],p[:,1]
        return r,*[self.predict(y,f,torch.full((len(y),),q,device=y.device)) for q in self.levels]
