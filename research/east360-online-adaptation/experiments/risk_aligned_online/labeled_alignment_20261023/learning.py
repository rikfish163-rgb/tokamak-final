"""Label-conditional feature moments, with an unchanged source risk head.

Mean matching follows the first/second-moment rationale of TTT++; covariance
uses Deep CORAL scaling. Conditioning on observed training labels is inspired
by S-CORAL, not an exact reproduction of its source-data transform + DART.
"""
import numpy as np
import torch

from ..deepcoral_prefix_20261022.learning import coral_loss
from .. import simple_data


def moment_loss(source, target):
    return (source.mean(0)-target.mean(0)).square().mean()+coral_loss(source,target)


def conditional_moment_loss(source, target, source_labels, target_labels):
    if source_labels.shape != (len(source),) or target_labels.shape != (len(target),):
        raise ValueError('one actual training label per feature row required')
    if not all(torch.all((y==0)|(y==1)) for y in (source_labels,target_labels)):
        raise ValueError('binary training labels required')
    terms = []
    for c in (0,1):
        s,t = source[source_labels==c],target[target_labels==c]
        if min(len(s),len(t)) < 2:
            raise ValueError('at least two actual samples per class/domain required')
        terms.append(moment_loss(s,t))
    return torch.stack(terms).mean()


def choose_budget(records, budget, *, test_ids):
    if budget not in (10,20) or len({r['shot'] for r in records}) != len(records):
        raise ValueError('distinct historical records and budget10/20 required')
    if {r['shot'] for r in records} & set(test_ids):
        raise ValueError('training/test overlap')
    if any(bool(r['is_disrupt']) != bool(r['shotlist_is_disrupt']) for r in records):
        raise ValueError('HDF5 and shotlist labels disagree; resolve separately')
    npositive = budget//5
    selected = []
    for label,count in ((True,npositive),(False,budget-npositive)):
        group = sorted((r for r in records if bool(r['is_disrupt'])==label),key=lambda r:r['shot'])
        if len(group) < count:
            raise ValueError('insufficient actual historical shots in class')
        selected.extend(group[:count])
    return sorted(selected,key=lambda r:r['shot'])


def balanced_batch(shots, groups, rng, cfg, *, per_class=16, device='cuda'):
    examples, receipts = [], []
    for label in (0,1):
        for _ in range(per_class):
            i,ends = groups[label][int(rng.integers(len(groups[label])))]
            end = int(rng.choice(ends)); s=shots[i]
            x,m = simple_data.window(s,end,cfg.window)
            examples.append((x,m,np.float32(label)))
            receipts.append(dict(shot=int(s['shot']),end=end,label=label))
    return (*[torch.as_tensor(np.stack(v),device=device) for v in zip(*examples)],receipts)
