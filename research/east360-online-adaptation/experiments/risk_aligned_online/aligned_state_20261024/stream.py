"""Within-shot encoder memory: predict/seal a block, update for later points."""
import numpy as np
import torch

from ..simple_data import window
from .model import update_fast

ARMS = ('off', 'real', 'constant', 'reverse')


def predict_stream(model, signal, *, device='cuda', interval=32, learning_rate=1e-4):
    if set(signal) != {'values','available'} or interval < 1:
        raise ValueError('signals only and positive update interval required')
    model.eval()
    curves = {a:[] for a in ARMS}; states = {a:model.new_fast() for a in ARMS if a != 'off'}
    records = []; n = len(signal['values'])
    for lo in range(0,n,interval):
        hi = min(lo+interval,n)
        windows = [window(signal,k,model.cfg.window) for k in range(lo,hi)]
        x,m = [torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
        with torch.no_grad():
            curves['off'].append(model(x,m)['logit'].sigmoid().cpu().numpy())
            for a in states:
                curves[a].append(model.predict_fast(x,m,states[a])['logit'].sigmoid().cpu().numpy())
        # No update changes any score in the block just sealed. Only full
        # arrived windows supply feedback; no terminal label is an argument.
        if hi >= model.cfg.window and hi < n:
            observed, available = x[-1:],m[-1:]
            valid = available & torch.isfinite(observed)
            count = valid.sum(1,keepdim=True)
            mean = torch.where(valid,observed,0.).sum(1,keepdim=True)/count.clamp_min(1)
            constant = torch.where(valid,mean.expand_as(observed),observed)
            for a,xx,mm in (('real',observed,available),('constant',constant,available),
                             ('reverse',observed.flip(1),available.flip(1))):
                states[a], info = update_fast(model,states[a],xx,mm,learning_rate=learning_rate)
                records.append(dict(arm=a,observed_end=hi-1,first_affected_index=hi,**info))
    result = {a:np.concatenate(v) for a,v in curves.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('nonfinite shot prediction')
    return result,records


@torch.no_grad()
def predict_static(model,signal,device='cuda',batch_size=32):
    values = []
    for start in range(0,len(signal['values']),batch_size):
        w = [window(signal,k,model.cfg.window) for k in range(start,min(start+batch_size,len(signal['values'])))]
        x,m = [torch.as_tensor(np.stack(a),device=device) for a in zip(*w)]
        values.append(model(x,m)['logit'].sigmoid().cpu().numpy())
    return np.concatenate(values)
