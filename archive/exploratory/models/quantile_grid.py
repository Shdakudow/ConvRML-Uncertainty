"""Matched decoder trunks for conditioned and simultaneous quantile prediction."""
import copy
import torch
from torch import nn

LEVELS=(.05,.25,.5,.75,.95)

def fresh_head(reconstruction_head):
    head=copy.deepcopy(reconstruction_head)
    for i,layer in enumerate(head.f):
        if isinstance(layer,nn.BatchNorm2d):
            head.f[i]=nn.GroupNorm(8,layer.num_features)
    return head.requires_grad_(True)

class GridHead(nn.Module):
    def __init__(self,template):
        super().__init__()
        self.head=copy.deepcopy(template)
        last=self.head.f[-1]
        self.head.f[-1]=nn.Conv2d(last.in_channels,15,3,padding=1,bias=False)
        with torch.no_grad():self.head.f[-1].weight.copy_(last.weight.repeat(5,1,1,1))
        self.requires_grad_(True)

    def forward(self,f):
        p=self.head(f)
        return p.reshape(len(p),5,3,*p.shape[-2:])

def pinball(pred,x,levels=LEVELS):
    q=pred.new_tensor(levels).reshape(1,-1,1,1,1)
    e=x[:,None]-pred
    return torch.maximum(q*e,(q-1)*e).mean()
