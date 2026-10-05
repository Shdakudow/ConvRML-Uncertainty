"""RGB adapters around the original U-Net; no ConvNeXt layers."""
import torch
from torch import nn
from .quantile_uqnet import UNetModel

def backbone(out_channels=3,dropout=0.,attention=True):
 return UNetModel(in_channels=3,out_channels=out_channels,attention_resolutions=tuple(256//r for r in (16,8,4,4,2,1)) if attention else (),dropout=dropout)

class OriginalQUTCC(nn.Module):
 def __init__(self):
  super().__init__();self.base=backbone()
 def quantile(self,y,q):return self.base(y,q)
 def forward(self,y):return self.quantile(y,y.new_full((len(y),),.5))
 def bounds(self,y):return self(y),self.quantile(y,y.new_full((len(y),),.05)),self.quantile(y,y.new_full((len(y),),.95))

class ConstantQUNet(nn.Module):
 def __init__(self,out_channels=3,dropout=0.):
  super().__init__();self.base=backbone(out_channels,dropout)
 def forward(self,y):return self.base(y,y.new_full((len(y),),.5))

class OriginalIm2Im(nn.Module):
 def __init__(self):
  super().__init__();self.base=backbone(32,attention=False)
  self.lower=nn.Conv2d(32,3,3,padding=1);self.prediction=nn.Conv2d(32,3,3,padding=1);self.upper=nn.Conv2d(32,3,3,padding=1)
 def bounds(self,y):
  f=self.base(y,y.new_full((len(y),),.5));return self.prediction(f),self.lower(f),self.upper(f)
 def forward(self,y):return self.bounds(y)[0]
