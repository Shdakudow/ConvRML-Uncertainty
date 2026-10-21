"""QUTCC-style scale/shift after normalization in ConvNeXt and its FPN.

The backbone is still Shamus ConvNeXtV2-Base. Hooks modulate the internal
residual branch (not the completed residual block). Sequential forwards only.
"""
from contextlib import contextmanager
import copy
import torch
from torch import nn
from models.shamus_rml import ShamusRML
from baselines.formal_rml import FormalRML
from models.qutcc_original.nn import timestep_embedding

class NormFiLM(nn.Module):
 def __init__(self,channels,channels_last=False):
  super().__init__();self.channels_last=channels_last
  self.projection=nn.Sequential(nn.SiLU(),nn.Linear(512,2*channels))
  # Preserve the initialized reconstruction before FiLM learns its modulation.
  nn.init.zeros_(self.projection[-1].weight);nn.init.zeros_(self.projection[-1].bias)
 def forward(self,x,emb):
  scale,shift=self.projection(emb).chunk(2,1)
  shape=(len(x),1,1,-1) if self.channels_last else (len(x),-1,1,1)
  return x*(1+scale.reshape(shape))+shift.reshape(shape)

class FiLMBackbone(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  self.embedding=nn.Sequential(nn.Linear(128,512),nn.SiLU(),nn.Linear(512,512))
  self.films=nn.ModuleDict();self.locations=[];self._emb=None;self._handles=[]
  for name,module in list(base.named_modules()):
   target=None;channels_last=False
   if module.__class__.__name__=='ConvNeXtBlock':
    target=module.norm;channels=module.conv_dw.out_channels;channels_last=not module.use_conv_mlp;site=name+'.norm'
   elif isinstance(module,nn.GroupNorm) and name.startswith(('refine.','up1.','up2.')):
    target=module;channels=module.num_channels;site=name
   if target is not None:
    key=site.replace('.','__');self.films[key]=NormFiLM(channels,channels_last);self.locations.append(dict(module=site,channels=channels,channels_last=channels_last))
    self._handles.append(target.register_forward_hook(self._hook(key)))
  if len(self.films)!=48:raise ValueError(f'Expected 36 encoder + 12 decoder sites, got {len(self.films)}')
 def _hook(self,key):
  def hook(module,inputs,output):
   if self._emb is None:raise RuntimeError('Backbone called outside FiLM context')
   return self.films[key](output,self._emb)
  return hook
 @contextmanager
 def condition(self,y,q=None):
  if self._emb is not None:raise RuntimeError('Concurrent/nested forwards are unsupported')
  if q is None:q=y.new_full((len(y),),.5)
  if q.shape!=(len(y),):raise ValueError('One q per image required')
  self._emb=self.embedding(timestep_embedding(q,128))
  try:yield
  finally:self._emb=None
 def forward(self,y,q=None):
  with self.condition(y,q):return self.base(y)
 def features(self,y,q=None):
  with self.condition(y,q):return self.base.up2(self.base.features(y))
 @property
 def head(self):return self.base.head

class ConvNeXtFiLMQuantile(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.base=FiLMBackbone(ShamusRML());self.levels=(.05,.95)
  if kind=='fixed':
   self.base.requires_grad_(False).eval();self.head=nn.Conv2d(256,6,3,padding=1)
   with torch.no_grad():self.head.weight.copy_(self.base.head.weight.repeat(2,1,1,1));self.head.bias.copy_(self.base.head.bias.repeat(2))
  elif kind=='conditioned':self.head=copy.deepcopy(self.base.head)
  else:raise ValueError(kind)
 def train(self,mode=True):
  super().train(mode)
  if self.kind=='fixed':self.base.eval()
  return self
 def features(self,y):
  f=self.base.features(y);return f,y+self.base.head(f)
 def predict(self,y,f=None,q=None):
  if self.kind=='fixed':return self.head(f).reshape(len(y),2,3,*y.shape[-2:])+y[:,None]
  return y+self.head(self.base.features(y,q))
 def bounds(self,y):
  if self.kind=='fixed':
   f,r=self.features(y);p=self.predict(y,f);return r,p[:,0],p[:,1]
  return self.base(y),self.predict(y,q=y.new_full((len(y),),.05)),self.predict(y,q=y.new_full((len(y),),.95))
