"""Per-shot rank-constrained attention updates; no trainable risk correction."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend


class AttentionAdapter(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        cfg=base.cfg
        if (cfg.dim,cfg.heads,cfg.layers,cfg.window,cfg.patch,cfg.horizon)!=(64,4,2,128,8,25) or cfg.input_view!='source':
            raise ValueError('locked W128/P8/d64/H25 source required')
        if any(not layer.norm_first or layer.dropout.p!=0 for layer in base.encoder.layers):
            raise ValueError('pre-norm dropout-free source required')
        rng=torch.Generator(device='cpu').manual_seed(131)
        for name,rank in (('q',2),('k',2),('v',4)):
            q,_=torch.linalg.qr(torch.randn(64,rank,generator=rng,dtype=torch.float64))
            self.register_buffer(name+'_basis',q.T.float().to(base.center.device))
        self.requires_grad_(False);self.eval()

    def deltas(self,state,mode):
        if state.ndim!=2 or state.shape[1]!=512 or not torch.isfinite(state).all() or mode not in ('qk','v'):
            raise ValueError('finite Bx512 temporary parameters and declared site required')
        names=('q','k') if mode=='qk' else ('v',);rank=2 if mode=='qk' else 4
        parts=state.reshape(len(state),len(names),2,rank,64)
        # State packs dA and B^T. A0 has orthonormal rows; B=0 at reset.
        return {name:parts[:,i,1].transpose(1,2)@(getattr(self,name+'_basis')+parts[:,i,0])
                for i,name in enumerate(names)}

    def first(self,values,available,state,mode):
        if values.shape!=available.shape or values.shape[1:]!=(128,23) or available.dtype!=torch.bool or len(state)!=len(values):
            raise ValueError('matching causal windows, masks and per-example states required')
        base=self.base;valid=available&torch.isfinite(values)
        z=torch.asinh((torch.where(valid,values,base.center)-base.center)/base.scale)
        x=base.embedding(torch.cat((z,valid.to(z.dtype)),-1).reshape(len(z),16,-1))
        x=torch.cat((x,base.cls.expand(len(x),-1,-1)),1)+base.position
        v=valid.reshape(len(x),16,-1).any(-1)
        v=torch.cat((v,torch.ones(len(x),1,device=x.device,dtype=torch.bool)),1)
        layer=base.encoder.layers[0];n=layer.norm1(x)
        packed=F.linear(n,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias)
        q,k,value=packed.chunk(3,-1);ds=self.deltas(state,mode)
        if mode=='qk':
            q=q+torch.bmm(n,ds['q'].transpose(1,2));k=k+torch.bmm(n,ds['k'].transpose(1,2))
        else:
            value=value+torch.bmm(n,ds['v'].transpose(1,2))
        project=lambda t:t.reshape(len(x),17,4,16).transpose(1,2)
        with sdpa_kernel(SDPBackend.MATH):
            y=F.scaled_dot_product_attention(project(q),project(k),project(value),attn_mask=v[:,None,None,:],dropout_p=0.)
        y=y.transpose(1,2).reshape(len(x),17,64)
        h=x+layer.dropout1(layer.self_attn.out_proj(y))
        h=h+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(h))))))
        return h,dict(query=q,key=k,value=value),v

    def forward(self,values,available,state,slow,mode,*,forecast=True):
        h,_,v=self.first(values,available,state,mode)
        if slow is not None:
            last=h[:,-2]+F.linear(h[:,-2],slow)*v[:,-2,None]
            h=torch.cat((h[:,:-2],last[:,None],h[:,-1:]),1)
        with sdpa_kernel(SDPBackend.MATH):
            h=self.base.encoder.layers[1](h,src_key_padding_mask=~v)
        gain,bias=self.base.fast_origin.chunk(2,-1)
        h=self.base.norm(h)*(1+.1*gain[None,None].tanh())+.1*bias[None,None].tanh()
        h=h.masked_fill(~v[...,None],0.)
        result=dict(logit=self.base.risk_head(h[:,-1]).squeeze(-1))
        if forecast:
            result['forecast']=self.base.forecast_head(h.flatten(1)).reshape(len(h),25,23)
        return result
