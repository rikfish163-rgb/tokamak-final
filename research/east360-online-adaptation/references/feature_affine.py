"""Learnable flat23 coordinates before the original source patch embedding.

This numerical feature adapter is not certified physical-unit calibration.
The original classifier and direct H25 head are frozen. Both tasks see the
same adapted inputs; there is no final-logit correction or extra output head.
Default persistent/disposable affine deltas have 46 parameters. The explicit
window_gain hypothesis instead has 23 disposable gains around the observed
past-window mean; persistent experience remains a 46-parameter affine.
"""
import math

import torch
from torch import nn

from .simple23 import Simple23


class FeatureAffineModel(nn.Module):
    temporary_dim=46
    adaptation_site='normalized flat23 before embedding'

    def __init__(self, base, *, forecast_coordinates='source', temporary_mode='affine'):
        super().__init__()
        if forecast_coordinates not in ('source','observed'):
            raise ValueError('explicit source or observed forecast coordinates required')
        if temporary_mode not in ('affine','window_gain'):
            raise ValueError('explicit affine or window_gain temporary mode required')
        if temporary_mode=='window_gain' and forecast_coordinates!='observed':
            raise ValueError('window_gain requires observed forecast coordinates')
        if not isinstance(base,Simple23) or (
                base.cfg.window,base.cfg.patch,base.cfg.dim,base.cfg.horizon,
                base.cfg.fast_site,base.cfg.input_view)!=(128,8,64,25,'tokens','source'):
            raise ValueError('qualified source-view W128/P8/d64/H25 model required')
        self.base=base.eval().requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.affine=nn.Parameter(base.center.new_zeros(46))
        self.forecast_coordinates=forecast_coordinates
        self.temporary_mode=temporary_mode
        if temporary_mode=='window_gain':
            self.temporary_dim=23
            self.adaptation_site='past-window variation before embedding; mean retained'

    @property
    def cfg(self):return self.base.cfg

    @property
    def center(self):return self.base.center

    @property
    def scale(self):return self.base.scale

    def train(self,mode=True):
        super().train(mode); self.base.eval(); return self

    def initial_fast(self,batch):
        if type(batch) is not int or batch<0:raise ValueError('nonnegative batch required')
        return self.affine.new_zeros(batch,self.temporary_dim)

    @staticmethod
    def coefficients(delta):
        if delta.ndim!=2 or delta.shape[1]!=46 or not torch.isfinite(delta).all():
            raise ValueError('finite [B,46] feature deltas required')
        gain,bias=delta.chunk(2,-1)
        # Numerical bounds: at most x2 or /2 in source-asinh coordinates,
        # and at most +/-2 additive asinh units. Not physical safety bounds.
        return (math.log(2)*gain.tanh()).exp(),2*bias.tanh()

    def forward(self,values,available,fast=None):
        if (values.ndim!=3 or values.shape[1:]!=(128,23) or not len(values)
                or values.dtype!=self.affine.dtype or values.device!=self.affine.device):
            raise ValueError('source dtype/device nonempty [B,128,23] required')
        if (available.shape!=values.shape or available.dtype!=torch.bool
                or available.device!=values.device):
            raise ValueError('matching boolean availability required')
        fast=self.initial_fast(len(values)) if fast is None else fast
        if (fast.shape!=(len(values),self.temporary_dim) or fast.dtype!=values.dtype
                or fast.device!=values.device or not torch.isfinite(fast).all()):
            raise ValueError(f'finite source dtype/device [B,{self.temporary_dim}] temporary state required')
        valid=available & torch.isfinite(values)
        clean=torch.where(valid,values,self.center)
        level=torch.asinh((clean-self.center)/self.scale)
        if self.temporary_mode=='affine':
            gain,bias=self.coefficients(fast+self.affine[None])
        else:
            gain,bias=self.coefficients(self.affine[None].expand(len(values),-1))
            count=valid.sum(1,keepdim=True)
            # First-valid anchoring keeps a constant channel EXACTLY constant,
            # avoiding cancellation from a float32 sum of repeated levels.
            anchor=level.gather(1,valid.to(torch.int64).argmax(1)[:,None])
            residual=torch.where(valid,level-anchor,0.)
            mean=(anchor+residual.sum(1,keepdim=True)/count.clamp_min(1)).detach()
            fast_gain=(math.log(2)*fast.tanh()).exp()
            # An entirely unseen channel cannot identify its local variability;
            # identity is explicit, tested, and applies to the forecast inverse.
            fast_gain=torch.where(count[:,0]>0,fast_gain,torch.ones_like(fast_gain))
        changed=level*gain[:,None]+bias[:,None]
        if self.temporary_mode=='window_gain':
            # Retain the observed operating level. Only temporal departures
            # from that SAME causal window mean are rescaled; no fast offset.
            changed=changed+gain[:,None]*(fast_gain[:,None]-1)*(level-mean)
        # Difference form is an exact zero identity, including gradients at
        # zero; unlike round-tripping raw values through sinh(asinh(x)).
        mapped=torch.where(valid,clean+self.scale*(changed.sinh()-level.sinh()),clean)
        if not torch.isfinite(mapped).all():raise FloatingPointError('nonfinite feature mapping')
        out=self.base(mapped,valid,self.base.initial_fast(len(values)))
        if self.forecast_coordinates=='observed':
            # The base predicts the adapted/source-asinh signal coordinate.
            # Undo exactly the input affine for the DIRECT forecast only, so
            # it can be compared with matured observations in their unchanged
            # coordinate. Risk still uses the adapted internal representation.
            # This is inspired by reversible forecasting, not fitted RevIN
            # instance statistics or a claim of physical-unit identification.
            forecast=(out['forecast']-bias[:,None])/gain[:,None]
            if self.temporary_mode=='window_gain':
                # Undo persistent affine, THEN undo the local variation gain.
                # Difference form is exact identity when the fast gain is one.
                forecast=forecast+(forecast-mean)*(fast_gain[:,None].reciprocal()-1)
            out=dict(out,forecast=forecast)
        return out

    def forecast_loss(self,prediction,truth,available):
        # Truth is never transformed by the temporary adapter. In reversible
        # mode only the PREDICTION is mapped back; observed truth stays fixed.
        return self.base.forecast_loss(prediction,truth,available)

    def persistent_parameters(self):return iter((self.affine,))
