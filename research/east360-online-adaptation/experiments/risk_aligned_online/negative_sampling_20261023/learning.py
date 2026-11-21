"""Matched target sampling; never relabel uncertain disruptive prefixes."""
import numpy as np
import torch

from .. import simple_data
from ..labeled_alignment_20261023.learning import balanced_batch

POLICIES=('legacy','safe_negative')


def paired_batch(shots,groups,rng,replacement_rng,cfg,policy,*,device='cuda'):
    if policy not in POLICIES or not shots or any(s['device']!='EAST' for s in shots):
        raise ValueError('explicit EAST preparation sampling policy required')
    safe=[(i,ends) for i,ends in groups[0] if not shots[i]['is_disrupt']]
    if not safe:raise ValueError('observed safe training shots required')
    x,m,y,receipts=balanced_batch(shots,groups,rng,cfg,device=device)
    proposals=[dict(r) for r in receipts];replacements=[];by_id={s['shot']:s for s in shots}
    if policy=='safe_negative':
        for j,row in enumerate(proposals):
            if row['label']!=0 or not by_id[row['shot']]['is_disrupt']:continue
            i,ends=safe[int(replacement_rng.integers(len(safe)))];s=shots[i]
            end=int(replacement_rng.choice(ends));xx,mm=simple_data.window(s,end,cfg.window)
            x[j]=torch.as_tensor(xx,device=device);m[j]=torch.as_tensor(mm,device=device)
            receipts[j]=dict(shot=int(s['shot']),end=end,label=0)
            replacements.append(dict(index=j,before=row,after=receipts[j]))
    return x,m,y,receipts,dict(proposed=proposals,replaced=replacements)
