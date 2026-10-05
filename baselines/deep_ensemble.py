"""Adapted from Downloads/deep_ensemble.py; RGB Gaussian-NLL members."""
import torch
from torch import nn
from torch.nn import functional as F
from .common import aggregate_moments,gaussian_interval_from_mean_variance


class DeepEnsembleMember(nn.Module):
    def __init__(self,base_model,*,channels=3,min_variance=1e-6):
        super().__init__();self.baseModel=base_model;self.channels=channels;self.min_variance=min_variance
        if min_variance<=0:raise ValueError('min_variance must be positive')

    def forward(self,x):
        raw=self.baseModel(x)
        if raw.shape[1]!=2*self.channels:raise ValueError('Member must output C means and C raw variances')
        return raw[:,:self.channels],F.softplus(raw[:,self.channels:])+self.min_variance

    def loss(self,x,target):
        mean,variance=self(x)
        return .5*(variance.log()+(target-mean).square()/variance).mean()


class DeepEnsemble(nn.Module):
    def __init__(self,members,*,base_interval_alpha=.1):
        super().__init__();self.members=nn.ModuleList(members);self.base_interval_alpha=base_interval_alpha
        if not len(self.members):raise ValueError('Ensemble cannot be empty')

    def moments(self,x):
        predictions=[m(x) for m in self.members]
        mean,total,_,_=aggregate_moments(torch.stack([v[0] for v in predictions]),torch.stack([v[1] for v in predictions]))
        return mean,total

    def predict_mean(self,x):return self.moments(x)[0]

    def forward(self,x):
        mean,variance=self.moments(x)
        lo,hi=gaussian_interval_from_mean_variance(mean,variance,self.base_interval_alpha)
        return torch.cat([lo,mean,hi],dim=1)
