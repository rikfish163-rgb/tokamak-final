"""Historical preparation and strictly window-local, outcome-free inference."""
import numpy as np
import torch
from torch.nn import functional as F

from ..simple_data import window
from ..matured_negative_20261023.learning import tensors

ARMS=('parent','capacity','adaptive_off','adaptive_on','adaptive_reverse')


def training_pools(records):
    pools={0:[],1:[]}
    for index,s in enumerate(records):
        eligible=np.asarray(s['valid']).copy()&(np.arange(len(s['times']))>=127)
        label=int(s['is_disrupt'])
        if label:
            lead=np.round((s['down']-s['times'])*1000,9)
            eligible&=(lead>30)&(lead<=125)
        ends=np.flatnonzero(eligible)
        if len(ends):pools[label].append((index,ends))
    if not all(pools.values()):raise ValueError('preparation needs safe and timely disruptive windows')
    return pools


def sample_batch(records,pools,rng,device):
    xs,ms,ys,receipt=[],[],[],[]
    for label in (0,1):
        for _ in range(16):
            index,ends=pools[label][int(rng.integers(len(pools[label])))];s=records[index]
            end=int(rng.choice(ends));x,m=window(s,end,128)
            xs.append(x);ms.append(m);ys.append(label)
            receipt.append(dict(shot=s['shot'],end=end,label=label))
    return (torch.as_tensor(np.stack(xs),device=device),torch.as_tensor(np.stack(ms),device=device),
            torch.tensor(ys,dtype=torch.float32,device=device),receipt)


def risk_step(model,optimizer,h,valid,y,*,mode):
    model.eval();optimizer.zero_grad(set_to_none=True)
    z,_=model.from_tokens(h.detach(),valid,mode=mode)
    loss=F.binary_cross_entropy_with_logits(z,y)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite preparation risk loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.layer.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),inner_rate=float(model.layer.eta.detach()))


@torch.no_grad()
def predict(adaptive,capacity,signal,*,device='cuda',batch=32):
    # Keep the archived parent's endpoint batch32. The real 71265 regression
    # was exact at32 but differed by5.3e-6 at256; changing MHA fastpath instead
    # increased the error to7.2e-4. Do not relax tolerances or alter the backend.
    if set(signal)!={'values','available'}:raise ValueError('observed signals only, no outcome input')
    adaptive.eval();capacity.eval();curves={a:[] for a in ARMS};traces={}
    for lo in range(0,len(signal['values']),batch):
        x,m=tensors(signal,range(lo,min(lo+batch,len(signal['values']))),device)
        h,valid=adaptive.tokens(x,m)
        for arm,model,mode in (('parent',adaptive,'bypass'),('capacity',capacity,'off'),
                ('adaptive_off',adaptive,'off'),('adaptive_on',adaptive,'on'),('adaptive_reverse',adaptive,'reverse')):
            z,info=model.from_tokens(h,valid,mode=mode)
            curves[arm].append(z.sigmoid().cpu().numpy())
            if arm=='adaptive_on':
                for name,value in info.items():traces.setdefault(name,[]).append(value.cpu().numpy())
    curves={a:np.concatenate(rows) for a,rows in curves.items()}
    if not all(np.isfinite(p).all() for p in curves.values()):raise FloatingPointError('nonfinite risk probability')
    return curves,{name:np.concatenate(rows) for name,rows in traces.items()}
