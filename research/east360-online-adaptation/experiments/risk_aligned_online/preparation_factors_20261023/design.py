"""Known-class historical preparation, never label-blind active acquisition."""
import numpy as np
import torch

from ..diverse_support_20261023.selection import select


def class_aware_selection(ids, descriptors, labels):
    ids=np.asarray(ids);x=np.asarray(descriptors)
    if (ids.ndim!=1 or len(set(ids))!=len(ids) or x.ndim!=2 or len(x)!=len(ids)
            or set(labels)!=set(ids) or not all(type(v) is bool for v in labels.values())):
        raise ValueError('exact candidate-only class map and matched descriptors required')
    chosen=[]
    for label,budget in ((True,4),(False,16)):
        use=np.array([labels[int(s)]==label for s in ids])
        if use.sum()<budget:raise ValueError('candidate pool does not meet the declared 4D/16S quota')
        chosen.extend(select(ids[use].tolist(),x[use],budget,'diverse',131))
    return sorted(chosen)


def risk_objective(source, target, use_source):
    if type(use_source) is not bool:raise ValueError('explicit source-supervision switch required')
    return .5*(source+target) if use_source else .5*target


def gradient_relation(source, target, parameters):
    parameters=tuple(parameters)
    gs=torch.autograd.grad(source,parameters,retain_graph=True,allow_unused=False)
    gt=torch.autograd.grad(target,parameters,retain_graph=True,allow_unused=False)
    a=torch.cat([g.detach().double().flatten() for g in gs])
    b=torch.cat([g.detach().double().flatten() for g in gt])
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError('nonfinite supervised gradient')
    na,nb=a.norm(),b.norm();zero=bool(na==0 or nb==0)
    return dict(source_norm=float(na),target_norm=float(nb),dot=float(a@b),
                cosine=None if zero else float((a@b)/(na*nb)),zero_gradient=zero)
