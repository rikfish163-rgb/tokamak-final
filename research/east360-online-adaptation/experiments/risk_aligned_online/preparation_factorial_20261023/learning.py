"""Two preparation factors; identical, strictly arrived intra-shot feedback."""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import adapt_arrived
from ..history_tangent_20261023.learning import logit
from ..labeled_alignment_20261023.learning import conditional_moment_loss
from ..paired_feedback_20261022.model import initial_encoder
from ..warmstart_alignment_20261022.alignment import fit_statistics,make_map,SIGNAL_KEYS

CONDITIONS=('raw_bce','raw_moments','mapped_bce','mapped_moments')


def mapping_for(source_stats,target,fit_ids,condition):
    if condition not in CONDITIONS:raise ValueError('explicit factorial condition required')
    if len(fit_ids)!=15 or len(set(fit_ids))!=15 or [r['shot'] for r in target]!=list(fit_ids):
        raise ValueError('exact fifteen-shot fitting set required')
    stats=fit_statistics([{k:r[k] for k in SIGNAL_KEYS} for r in target],device='EAST',allowed_ids=fit_ids)
    mapping=make_map(source_stats,stats,'density_gain')
    if condition.startswith('raw'):
        # Explicit no-empirical-mapping intervention, NOT a physical unit claim.
        # Necessary causal_input handling has already occurred in both arms.
        mapping['policy']='no_empirical_density_gain'
        mapping['gain']=[1.]*23;mapping['offset']=[0.]*23
        mapping['channels']=[dict(name=c['name'],status='declared_identity_control',gain=1.,offset=0.) for c in mapping['channels']]
    return mapping


def objective(head,hs,ht,ys,yt,condition):
    if condition not in CONDITIONS:raise ValueError('explicit factorial condition required')
    bs=F.binary_cross_entropy_with_logits(head(hs).squeeze(-1),ys)
    bt=F.binary_cross_entropy_with_logits(head(ht).squeeze(-1),yt)
    align=conditional_moment_loss(hs,ht,ys,yt)
    supervised=.5*(bs+bt)
    loss=supervised+align if condition.endswith('moments') else supervised
    return loss,dict(source_bce=float(bs.detach()),target_bce=float(bt.detach()),alignment=float(align.detach()))


def predict(model,signal,*,device='cuda'):
    if (set(signal)!={'values','available'} or signal['values'].ndim!=2
            or signal['values'].shape[1]!=23 or signal['available'].shape!=signal['values'].shape
            or signal['available'].dtype!=bool or model.cfg.window!=128 or model.cfg.horizon!=25):
        raise ValueError('signal-only W128 H25 flat23 stream required')
    n=len(signal['values'])
    if not n:raise ValueError('nonempty stream required')
    states={a:initial_encoder(model) for a in ('static','intra')}
    scores={a:[] for a in states};updates=[]
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,e,128) for e in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                for a in states:scores[a].append(logit(model,x,m,states[a]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        prefix={k:v[:hi] for k,v in signal.items()}
        states['intra'],receipt=adapt_arrived(model,prefix,hi,states['intra'],device=device)
        updates.append(receipt)
    result={a:np.concatenate(v) for a,v in scores.items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise FloatingPointError('nonfinite risk')
    return result,updates
