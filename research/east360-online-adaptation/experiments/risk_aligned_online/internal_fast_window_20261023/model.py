"""Task-trained, window-local fast matrix between two frozen Transformer blocks.

An independent masked least-squares TTT variant, not a reproduction of MedTS.
The inner step has a closed-form gradient; slow preparation differentiates
through it. No risk logit residual, full-shot statistics or outcome input.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend


class InnerLayer(nn.Module):
    def __init__(self,dim,rank=16):
        super().__init__()
        self.rank=rank
        self.query=nn.Linear(dim,rank,bias=False)
        self.key=nn.Linear(dim,rank,bias=False)
        self.value=nn.Linear(dim,rank,bias=False)
        self.origin=nn.Parameter(torch.eye(rank))
        self.rate=nn.Parameter(torch.tensor(0.))
        self.output=nn.Linear(rank,dim,bias=False)
        nn.init.zeros_(self.output.weight)

    @property
    def eta(self):return .5*self.rate.sigmoid()

    def coordinates(self,h):
        # Only the auxiliary coordinate path is normalized. The original
        # representation and physical amplitude pathway remain intact.
        u=F.layer_norm(h,(h.shape[-1],))
        return self.query(u).tanh(),self.key(u).tanh(),self.value(u).tanh()

    def fast_weights(self,h,valid,*,mode):
        if (mode not in ('on','off','reverse') or h.ndim!=3
                or valid.shape!=h.shape[:2] or valid.dtype!=torch.bool):
            raise ValueError('observed tokens with final CLS and declared mode required')
        _,k,v=self.coordinates(h)
        observed=valid.clone();observed[:,-1]=False
        count=observed.sum(-1);w=self.origin.expand(len(h),-1,-1)
        if mode=='reverse':
            # Reverse ONLY valid measured token positions, excluding CLS.
            length=h.shape[1];positions=torch.arange(length,device=h.device).expand(len(h),-1)
            ordered=torch.where(observed,positions,length).sort(-1).values
            ranks=observed.long().cumsum(-1)-1
            indices=ordered.gather(1,(count[:,None]-1-ranks).clamp(0,length-1)).clamp_max(length-1)
            v=v.gather(1,indices[:,:,None].expand_as(v))
        error=(k@w-v)*observed[:,:,None]
        gradient=k.transpose(1,2)@error/count[:,None,None].clamp_min(1)
        updated=(count>=2)&(mode!='off')
        fast=w-self.eta*gradient*updated[:,None,None]
        info=dict(updated=updated.detach(),observations=count.detach(),
            delta_norm=(fast-w).detach().square().sum((1,2)).sqrt(),
            loss_before=(error.square().sum((1,2))/(2*count.clamp_min(1))).detach(),
            loss_after=(((k@fast-v).square()*observed[:,:,None]).sum((1,2))/(2*count.clamp_min(1))).detach())
        return fast,info

    def forward(self,h,valid,*,mode):
        q,_,_=self.coordinates(h);fast,info=self.fast_weights(h,valid,mode=mode)
        # The zero output projection preserves the parent initially. Do NOT
        # also subtract I: with W0=I that double-zero initialization would
        # give the capacity/off control zero gradients and prevent learning.
        residual=self.output(q@fast)*valid[:,:,None]
        info['residual_norm']=residual.detach().square().sum((1,2)).sqrt()
        return h+residual,info


class WindowAdapter(nn.Module):
    def __init__(self,base):
        super().__init__()
        if base.cfg.layers!=2 or base.cfg.fast_site!='tokens' or base.cfg.input_view!='source':
            raise ValueError('qualified two-block token/source model required')
        self.base=base;self.cfg=base.cfg
        self.base.requires_grad_(False);self.base.eval()
        self.layer=InnerLayer(base.cfg.dim)

    def tokens(self,values,available):
        if (values.ndim!=3 or values.shape[1:]!=(self.cfg.window,23)
                or available.shape!=values.shape or available.dtype!=torch.bool):
            raise ValueError('flat23 observed past windows and matching availability required')
        mask=available&torch.isfinite(values)
        clean=torch.where(mask,values,self.base.center)
        level=torch.asinh((clean-self.base.center)/self.base.scale)
        x=torch.cat((level,mask.to(level.dtype)),-1)
        x=self.base.embedding(x.reshape(len(x),self.cfg.window//self.cfg.patch,-1))
        x=torch.cat((x,self.base.cls.expand(len(x),-1,-1)),1)+self.base.position
        valid=mask.reshape(len(x),self.cfg.window//self.cfg.patch,-1).any(-1)
        valid=torch.cat((valid,torch.ones(len(x),1,device=x.device,dtype=torch.bool)),1)
        with sdpa_kernel(SDPBackend.MATH):
            h=self.base.encoder.layers[0](x,src_key_padding_mask=~valid)
        return h,valid

    def from_tokens(self,h,valid,*,mode):
        if mode=='bypass':
            info=dict(updated=torch.zeros(len(h),device=h.device,dtype=torch.bool))
        else:h,info=self.layer(h,valid,mode=mode)
        with sdpa_kernel(SDPBackend.MATH):
            h=self.base.encoder.layers[1](h,src_key_padding_mask=~valid)
        gain,bias=self.base.initial_fast(len(h)).chunk(2,-1)
        h=self.base.norm(h)*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
        return self.base.risk_head(h[:,-1]).squeeze(-1),info

    def forward(self,values,available,*,mode='on'):
        h,valid=self.tokens(values,available)
        return self.from_tokens(h,valid,mode=mode)
