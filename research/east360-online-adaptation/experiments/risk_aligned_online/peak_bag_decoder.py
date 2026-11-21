"""Train the update direction at reference AND actually emitted ended peaks.

Finite two-endpoint MIL surrogate, not exhaustive whole-trajectory supervision.
The immutable source risk head is never trained. All bag construction is after
whole-shot seal and natural outcome reveal. Online FFN state remains disposable.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .continual_decoder import RollingDecoder
from .ended_context_rule import causal_peak_logit, peak_index
from .ended_decoder import maintenance_loss


def bag_warning_loss(logits, *, label, boundary):
    if (logits.ndim != 1 or not 1 <= len(logits) <= 2 or not torch.isfinite(logits).all()
            or label not in (0,1) or not math.isfinite(boundary)):
        raise ValueError('one/two finite logits, binary ended label and fixed boundary required')
    # A safe shot is wrong if ANY selected endpoint is high. A disruptive shot
    # only needs one eligible timely endpoint. amax uses the same maximum for
    # both objectives, without giving a two-endpoint task double the weight.
    return F.softplus((1-2*label)*(logits.amax()-boundary))


class BagRollingDecoder(RollingDecoder):
    def _prepare_shot(self, shot, *, phase, device):
        ended, reference = super()._prepare_shot(shot,phase=phase,device=device)
        emitted = peak_index(ended, ended['_sealed_scores'])
        return dict(ended,_bag_queries=tuple(sorted({reference,emitted}))), reference

    def _train_pair(self, *args, **kwargs):
        return train_bag_pair(*args, **kwargs)


def train_bag_pair(model, pair, optimizer, *, boundary, mode):
    if mode != 'risk_after_update' or not math.isfinite(boundary):
        raise ValueError('explicit post-adaptation risk objective required')
    allowed={id(p) for p in model.decoder.parameters()}
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    if (set(actual)!=allowed or len(actual)!=len(allowed)
            or {id(p) for p in model.parameters() if p.requires_grad}!=allowed):
        raise ValueError('only the ended-EAST auxiliary decoder may be optimized')
    if (len(pair)!=2 or sorted(int(s['is_disrupt']) for s,_ in pair)!=[0,1]
            or any(s.get('device')!='EAST' or '_sealed_scores' not in s
                   or s.get('horizon')!=125. for s,_ in pair)):
        raise ValueError('one balanced pair of sealed ended EAST shots required')
    # Validate the entire pair BEFORE any forward/backward/optimizer mutation.
    for shot, reference in pair:
        queries=shot.get('_bag_queries')
        if (not isinstance(queries,tuple) or not 1<=len(queries)<=2
                or any(type(q) is not int for q in queries)
                or tuple(sorted(set(queries)))!=queries or reference not in queries):
            raise ValueError('one/two distinct ordered reference/emitted endpoints required')
        emitted=peak_index(shot,shot['_sealed_scores'])
        if queries!=tuple(sorted({reference,emitted})):
            raise ValueError('bag differs from reference and actual sealed peak')
        for query in queries:
            if (not 0<=query<len(shot['times']) or not shot['valid'][query]
                    or shot['is_disrupt'] and
                    round(float((shot['down']-shot['times'][query])*1000.),9)<=30.):
                raise ValueError('training peak must be operationally eligible')
    base_before={k:v.detach().clone() for k,v in model.base.state_dict().items()}
    decoder_before=[p.detach().clone() for p in model.decoder.parameters()]
    model.zero_grad(set_to_none=True)
    records=[]; risk_gradient_square=0.
    for shot, reference in pair:
        logits=[]; infos=[]
        for query in shot['_bag_queries']:
            logit,info=causal_peak_logit(model,shot,query,create_graph=True)
            logits.append(logit);infos.append(dict(query=query,**info))
        stacked=torch.stack(logits)
        risk=bag_warning_loss(stacked,label=int(shot['is_disrupt']),boundary=boundary)
        signal=maintenance_loss(model,shot)
        if risk.requires_grad:
            gradients=torch.autograd.grad(risk,tuple(model.decoder.parameters()),
                                          retain_graph=True,allow_unused=True)
            risk_gradient_square+=sum(float(g.detach().double().square().sum())
                                      for g in gradients if g is not None)
        loss=risk+.20*signal
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite ended-head bag objective')
        (loss/2).backward()
        selected=int(stacked.detach().argmax())
        records.append(dict(shot=shot['shot'],label=int(shot['is_disrupt']),
            reference_query=reference,queries=list(shot['_bag_queries']),
            selected_query=shot['_bag_queries'][selected],
            adapted_probabilities=[float(z.detach().sigmoid()) for z in logits],
            causal_replays=infos,risk_loss=float(risk.detach()),
            maintenance_loss=float(signal.detach()),objective=float(loss.detach())))
    norm=torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.decoder.parameters()):
        raise FloatingPointError('nonfinite auxiliary decoder parameter')
    if model.memory.any() or model.memory.grad is not None:
        raise AssertionError('bag training changed persistent FFN')
    for key,value in model.base.state_dict().items():
        torch.testing.assert_close(value,base_before[key],rtol=0,atol=0)
    delta=sum(float((p.detach().double()-q.double()).square().sum())
              for p,q in zip(model.decoder.parameters(),decoder_before))**.5
    return dict(records=records,gradient_norm=float(norm),decoder_delta_norm=delta,
                risk_gradient_norm=math.sqrt(risk_gradient_square),
                objective=float(np.mean([r['objective'] for r in records])))
