"""Frozen-model temporal sensitivity controls, NOT physical interventions.

Window medians are available at the current endpoint. Removing within-window
variation preserves medians and masks but can create off-manifold trajectories;
resulting score differences do not identify plasma causality.
"""
import numpy as np
import torch

from ..paired_feedback_20261022.model import initial_encoder


def flatten_window(values,available,columns):
    x,m=np.asarray(values),np.asarray(available)
    columns=tuple(columns)
    if (x.ndim!=3 or x.shape[-1]!=23 or m.shape!=x.shape or m.dtype!=bool
            or not np.issubdtype(x.dtype,np.floating) or not np.isfinite(x[m]).all()
            or not columns or len(set(columns))!=len(columns)
            or any(type(c) is not int or not 0<=c<23 for c in columns)):
        raise ValueError('finite observed B,T,23 windows and distinct valid columns required')
    result=x.copy();ids=list(columns);observed=m[:,:,ids]
    selected=np.where(observed,x[:,:,ids],np.nan)
    empty=~observed.any(axis=1)
    # An entirely missing column stays missing; zero is only an internal
    # median-computation placeholder and is never written as an observation.
    selected[:,0,:]=np.where(empty,0.,selected[:,0,:])
    medians=np.nanmedian(selected,axis=1,keepdims=True)
    result[:,:,ids]=np.where(observed,medians,x[:,:,ids])
    return result


def causal_windows(signal,window):
    if set(signal)!={'values','available'} or type(window) is not int or window<1:
        raise ValueError('signal-only causal windows required')
    x,m=np.asarray(signal['values']),np.asarray(signal['available'])
    if (x.ndim!=2 or x.shape[-1]!=23 or not len(x) or m.shape!=x.shape
            or m.dtype!=bool or not np.isfinite(x[m]).all()):
        raise ValueError('finite observed N,23 stream and boolean availability required')
    return tuple(np.lib.stride_tricks.sliding_window_view(
        np.pad(a,((window-1,0),(0,0))),window,axis=0).swapaxes(1,2) for a in (x,m))


def predict_controls(model,signal,*,device='cuda'):
    windows,masks=causal_windows(signal,model.cfg.window)
    fast=initial_encoder(model);out={name:[] for name in ('original','density_flat','all_flat')}
    for lo in range(0,len(windows),32):
        x=np.array(windows[lo:lo+32],copy=True);m=np.array(masks[lo:lo+32],copy=True)
        tm=torch.as_tensor(m,device=device)
        for name,columns in (('original',None),('density_flat',(12,)),('all_flat',tuple(range(23)))):
            xx=x if columns is None else flatten_window(x,m,columns)
            h=model.predict_fast(torch.as_tensor(xx,device=device),tm,fast,mode='risk_state')['risk_state']
            out[name].append(model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().numpy())
    result={k:np.concatenate(v) for k,v in out.items()}
    if not all(v.shape==(len(windows),) and np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('incomplete/nonfinite diagnostic curves')
    return result
