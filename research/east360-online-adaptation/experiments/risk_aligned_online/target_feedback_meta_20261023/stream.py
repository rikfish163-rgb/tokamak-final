"""Same arrived H25 operator, computing the identical static trace only once."""
import numpy as np
import torch

from .. import simple_data
from ..paired_feedback_20261022.model import initial_encoder
from ..deepcoral_prefix_20261022.learning import adapt_arrived


def predict_five(models,signal,*,device='cuda'):
    if set(signal)!={'values','available'} or 'warm' not in models:
        raise ValueError('signal-only input and warm reference required')
    warm=models['warm'];state=initial_encoder(warm)
    fast={k:initial_encoder(m) for k,m in models.items()}
    predictions={k:[] for k in ['static']+[f'{k}_intra' for k in models]}
    updates={k:[] for k in models};n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,t,warm.cfg.window) for t in range(start,min(start+32,hi))]
            x,mask=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            for name,model,params in [('static',warm,state)]+[(f'{k}_intra',m,fast[k]) for k,m in models.items()]:
                h=model.predict_fast(x,mask,params,mode='risk_state')['risk_state']
                p=model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().numpy()
                predictions[name].append(p)
        for name,model in models.items():
            if hi<model.cfg.window+model.cfg.horizon or hi==n:continue
            fast[name],receipt=adapt_arrived(model,signal,hi,fast[name],device=device)
            updates[name].append(receipt)
    result={k:np.concatenate(v) for k,v in predictions.items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise FloatingPointError('nonfinite prediction')
    return result,updates
