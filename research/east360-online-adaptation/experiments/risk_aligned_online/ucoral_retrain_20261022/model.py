"""Existing classifier in explicitly standardized coordinates; no new risk head."""
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel,SDPBackend


def risk_from_coordinates(base,level,available):
    if base.cfg.fast_site!='tokens' or base.cfg.input_view!='source':
        raise ValueError('qualified source/token architecture required')
    if level.ndim!=3 or level.shape[1:]!=(base.cfg.window,23) or available.shape!=level.shape:
        raise ValueError('matching causal windows required')
    b=len(level);n=base.cfg.window//base.cfg.patch
    x=base.embedding(torch.cat((level,available.to(level.dtype)),-1).reshape(b,n,-1))
    valid=available.reshape(b,n,-1).any(-1)
    x=torch.cat((x,base.cls.expand(b,-1,-1)),1)+base.position
    valid=torch.cat((valid,torch.ones(b,1,dtype=torch.bool,device=x.device)),1)
    with sdpa_kernel(SDPBackend.MATH):x=base.encoder(x,src_key_padding_mask=~valid)
    gain,bias=base.initial_fast(b).chunk(2,-1)
    x=base.norm(x)*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
    x=x.masked_fill(~valid[:,:,None],0.)
    return base.risk_head(x[:,-1]).squeeze(-1)


class CoordinateClassifier(nn.Module):
    def __init__(self,base,fit,*,coral):
        super().__init__();self.base=base;self.cfg=base.cfg;self.coral=bool(coral)
        self.register_buffer('active',torch.tensor(fit['active'],dtype=torch.bool))
        for domain in ('source','target'):
            for key in ('mean','scale'):
                self.register_buffer(f'{domain}_{key}',torch.tensor(fit[domain][key],dtype=torch.float64))
        self.register_buffer('matrix',torch.tensor(fit['matrix'],dtype=torch.float64))
        base.requires_grad_(True);base.forecast_head.requires_grad_(False);base.fast_origin.requires_grad_(False)

    def transform(self,x,available,domain):
        if domain not in ('source','target'):raise ValueError('explicit domain required')
        if available.dtype!=torch.bool or available.shape!=x.shape:raise ValueError('boolean matching mask required')
        m=available & torch.isfinite(x) & self.active
        # Float64 normalization keeps small physical-file scales identifiable;
        # output is float32 for the unchanged classifier. Mask coordinates are
        # never linearly mixed or reused to erase mixed source values.
        z=torch.where(m,(x.double()-getattr(self,f'{domain}_mean'))/getattr(self,f'{domain}_scale'),0.)
        if domain=='source' and self.coral:z=z@self.matrix
        return z.to(x.dtype),m

    def forward(self,x,available,domain='source'):
        z,m=self.transform(x,available,domain)
        return dict(logit=risk_from_coordinates(self.base,z,m))
