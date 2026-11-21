"""Identical frozen-parent inputs; five full curves including exact off control."""
import numpy as np
import torch
from ..matured_negative_20261023.learning import tensors

ARMS=('parent','fixed_on','normalized_on','normalized_off','normalized_reverse')


@torch.no_grad()
def predict(fixed,normalized,signal,*,device='cuda',batch=32):
    if set(signal)!={'values','available'}: raise ValueError('outcome-free observed arrays only')
    fixed.eval(); normalized.eval(); curves={a:[] for a in ARMS}; traces={a:{} for a in ('fixed','normalized')}
    for lo in range(0,len(signal['values']),batch):
        x,m=tensors(signal,range(lo,min(lo+batch,len(signal['values']))),device)
        h,valid=normalized.tokens(x,m)
        for arm,model,mode in (('parent',normalized,'bypass'),('fixed_on',fixed,'on'),
                ('normalized_on',normalized,'on'),('normalized_off',normalized,'off'),('normalized_reverse',normalized,'reverse')):
            z,info=model.from_tokens(h,valid,mode=mode); curves[arm].append(z.sigmoid().cpu().numpy())
            if arm in ('fixed_on','normalized_on'):
                for k,v in info.items(): traces[arm.split('_')[0]].setdefault(k,[]).append(v.cpu().numpy())
    curves={a:np.concatenate(rows) for a,rows in curves.items()}
    if not all(np.isfinite(p).all() for p in curves.values()): raise FloatingPointError('nonfinite risk')
    np.testing.assert_array_equal(curves['parent'],curves['normalized_off'])
    return curves,{a:{k:np.concatenate(v) for k,v in row.items()} for a,row in traces.items()}
