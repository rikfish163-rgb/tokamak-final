"""Teach arrived-signal feedback using historical EAST labels, never TEST.

Persistent optimization is restricted to the auxiliary forecast head. The
temporary encoder update is exactly the previous deployed H25 operator.
The maximum objective covers one future prediction block, NOT a whole-shot
maximum; full-shot generalization must be measured on replayed trajectories.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..paired_feedback_20261022.model import initial_encoder
from ..matched_forecast_meta_20261022.learning import observed_step


def block_risk(logits,disrupt,boundary,objective):
    if logits.ndim!=1 or not len(logits) or not math.isfinite(boundary):
        raise ValueError('nonempty vector of logits and finite fixed boundary required')
    margin=logits-boundary
    if objective=='maximum':margin=margin.max()
    elif objective!='point':raise ValueError('point or maximum objective required')
    return F.softplus(-margin if disrupt else margin).mean()


def episode_pool(shots,allowed_ids):
    allowed=set(allowed_ids)
    if (len(allowed)!=len(allowed_ids) or len({s['shot'] for s in shots})!=len(shots)
            or any(s['device']!='EAST' or s['shot'] not in allowed for s in shots)):
        raise ValueError('distinct authorized historical EAST TRAIN identities required')
    rows=[]
    for s in shots:
        blocks=[]
        for count in range(256,len(s['values']),128):
            ends=np.arange(count,min(count+128,len(s['values'])))
            valid=s['valid'][ends].copy()
            if s['is_disrupt']:
                lead=np.round((s['down']-s['times'][ends])*1000.,9)
                valid &= (lead>30)&(lead<=125)&(s['y'][ends]>0)
            ends=ends[valid]
            if len(ends):blocks.append(dict(count=count,ends=ends.tolist()))
        rows.append(dict(shot=int(s['shot']),is_disrupt=bool(s['is_disrupt']),blocks=blocks,
                         exclusion=None if blocks else 'no_post256_eligible_prediction_block'))
    return rows


def block_episode(model,signal,count,ends,disrupt,*,boundary,objective,device,create_graph):
    if (set(signal)!={'values','available'} or type(count) is not int or count<256 or count%128
            or count>=len(signal['values']) or not ends or ends!=sorted(set(ends))
            or min(ends)<count or max(ends)>=min(count+128,len(signal['values']))):
        raise ValueError('signal-only input and query endpoints inside next fixed prediction block required')
    counts=list(range(256,count+1,128));tail=counts[-2:]
    fast=initial_encoder(model);losses=[]
    for seen in counts:
        fast,loss=observed_step(model,signal,seen,fast,device=device,
                               create_graph=create_graph and seen in tail)
        if seen in tail:losses.append(loss)
    windows=[simple_data.window(signal,end,model.cfg.window) for end in ends]
    x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
    z=model.predict_fast(x,m,fast,mode='risk_state')['risk_state']
    z=model.base.risk_head(z).squeeze(-1)
    risk=block_risk(z,disrupt,boundary,objective)
    initial=initial_encoder(model)
    h0=model.predict_fast(x,m,initial,mode='risk_state')['risk_state']
    before=block_risk(model.base.risk_head(h0).squeeze(-1),disrupt,boundary,objective).detach()
    return dict(risk_before=before,risk_after=risk,forecast=torch.stack(losses).mean(),
                all_update_counts=counts,differentiated_counts=tail,fast=fast)
