"""Same 9x32 forward geometry as archived cold; all Q/K states are zero."""
import numpy as np
import torch

from .. import simple_data
from .core import NAMES, apply_map, chunks


@torch.no_grad()
def predict(net, signal, slow, maps):
    if set(signal)!={'values','available'} or tuple(maps)!=NAMES:
        raise ValueError('signal-only input and all six fixed maps required')
    output = {n:np.empty(len(signal['values']),np.float32) for n in NAMES}
    state = net.base.center.new_zeros(9*32,8192)
    for start,stop in chunks(len(signal['values'])):
        pairs = [simple_data.window(signal,e,128) for e in range(start,stop)]
        x,m = [np.stack(v) for v in zip(*pairs)]
        count = len(x)
        if count<32:
            x = np.concatenate((x,np.repeat(x[-1:],32-count,axis=0)))
            m = np.concatenate((m,np.repeat(m[-1:],32-count,axis=0)))
        flat = dict(values=x.reshape(-1,23), available=m.reshape(-1,23))
        inputs = [apply_map(flat,maps[n])['values'].reshape(32,128,23) for n in NAMES]
        # Last three rows are computational padding only, not extra experiments.
        inputs.extend([inputs[0]]*3)
        values = torch.as_tensor(np.concatenate(inputs),device=net.base.center.device)
        masks = torch.as_tensor(np.tile(m,(9,1,1)),device=values.device)
        logits = net(values,masks,state,slow,'qk',forecast=False)['logit'].reshape(9,32)
        scores = torch.stack([row.sigmoid() for row in logits]).cpu().numpy()
        if not np.isfinite(scores).all() or not np.array_equal(scores[0],scores[5]):
            raise AssertionError('invalid scores or changed identity control')
        for i,name in enumerate(NAMES):
            output[name][start:stop] = scores[i,:count]
    return output
