"""Separate class balance from the disputed disruptive-prefix negatives."""
import numpy as np
import torch

from ..simple_data import window


def source_groups(shots,*,policy):
    if policy not in ('balanced','safe_negative'): raise ValueError('explicit source sampling policy required')
    groups={0:[],1:[]}
    for i,s in enumerate(shots):
        if s['device']!='J-TEXT': raise ValueError('source-only sampling')
        for y in (0,1):
            if policy=='safe_negative' and y==0 and s['is_disrupt']: continue
            ends=np.flatnonzero(s['valid'] & ((s['y']>.5)==bool(y)))
            if len(ends): groups[y].append((i,ends))
    if not all(groups.values()): raise ValueError('both source endpoint classes required')
    return groups


def balanced_pair(shots,groups,rng,cfg,device):
    examples=[];receipt=[]
    for y in (0,1):
        # Retain a 1/4 event-interval sampling opportunity WITHIN each class.
        # If that class has no annotated intervals, its ordinary valid pool
        # is the explicitly defined sampling population; unknown is never 0.
        event=[]
        for i,ends in groups[y]:
            selected=ends[shots[i]['event_target'][ends]>=0]
            if len(selected): event.append((i,selected))
        prefer_event=rng.random()<.25
        group=event if prefer_event and event else groups[y]
        i,ends=group[int(rng.integers(len(group)))];end=int(rng.choice(ends));s=shots[i]
        x,m=window(s,end,cfg.window);examples.append((x,m,float(y)))
        receipt.append(dict(shot=s['shot'],end=end,y=y,is_disrupt=bool(s['is_disrupt']),
                            event_known=bool(s['event_target'][end]>=0),event_pool_used=bool(prefer_event and event)))
    x,m,y=[torch.as_tensor(np.stack(v),device=device) for v in zip(*examples)]
    return x,m,y.float(),receipt
