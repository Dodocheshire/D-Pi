"""Pi at frozen RN50 layer4, before the unchanged attention pooling layer."""

import torch
from torch import nn

from .model import PiGenerator


class InternalDPi(nn.Module):
    def __init__(self,projection,weight,bias,pool,channel_scale,mode='mean_scale',
                 conditioning='concept',hidden=128,initial_scale=0.01):
        super().__init__()
        self.projection=projection
        self.pool=pool.eval().requires_grad_(False)
        self.mode,self.initial_scale=mode,initial_scale
        self.head=nn.Linear(len(projection.centers),len(bias))
        with torch.no_grad():
            self.head.weight.copy_(weight);self.head.bias.copy_(bias)
        self.head.requires_grad_(False)
        self.register_buffer('channel_scale',channel_scale.float())
        self.generator=None
        if mode in {'mean_scale','mean_only','adapter'}:
            self.generator=PiGenerator(projection.centers.shape[1],projection.centers,hidden,
                                       initial_scale,conditioning,mode,output_dim=len(channel_scale))

    def train(self,mode=True):
        super().train(mode)
        self.pool.eval()
        return self

    def forward(self,maps,clean_features,sample=True):
        clean=self.projection(clean_features)
        shape=(len(maps),len(self.channel_scale))
        if self.generator is None:
            mean=clean_features.new_zeros(shape)
            scale=torch.full_like(mean,self.initial_scale) if self.mode=='fixed' else torch.zeros_like(mean)
        else:mean,scale=self.generator.parameters_for(clean_features)
        noise=mean+scale*torch.randn_like(scale) if sample else mean
        shifted=maps.float()+self.channel_scale[None,:,None,None]*noise[:,:,None,None]
        pool_dtype=next(self.pool.parameters()).dtype
        features=self.pool(shifted.to(pool_dtype)).float()
        concepts=self.projection(features)
        return self.head(concepts),concepts,clean,(mean.square()+scale.square()).mean()
