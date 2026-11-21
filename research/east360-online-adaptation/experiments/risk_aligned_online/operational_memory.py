"""One ended-shot FFN proposal protected by actual complete causal replay.

No final-score head, alarm tuning, or new target scaler. The history guard is a
bounded engineering retention check, not GEM's quadratic program or a future
generalization guarantee. The gradient unroll is truncated to four mature
updates; the candidate's deployed trajectory is replayed in full.
"""
from copy import deepcopy
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data
from .continual_decoder import head_digest
from .ended_context_rule import causal_peak_logit
from .fast_update import normalized_step
from .redesign_20261004.intra import _batch_windows, predict_shot
from .redesign_20261004.source_replay import source_loss, evaluate_source_guard


def operational_state(shot, scores, threshold):
    """Whole-shot first alarm and current eligible maximum, strictly lead>30ms."""
    scores=np.asarray(scores)
    times=np.asarray(shot['times'])
    valid=np.asarray(shot['valid'],bool).copy()
    if (shot.get('device')!='EAST' or shot.get('horizon')!=125.
            or '_sealed_scores' not in shot or shot.get('is_disrupt') not in (0,1)
            or scores.shape!=times.shape or valid.shape!=times.shape
            or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0) or not np.isfinite(scores).all()
            or np.any((scores<0)|(scores>1))
            or not math.isfinite(threshold) or not 0<threshold<1):
        raise ValueError('finite complete sealed ended EAST trajectory and fixed threshold required')
    label=int(shot['is_disrupt'])
    if label:
        if not math.isfinite(shot['down']):
            raise ValueError('finite ended disruption time required')
        valid &= times<=shot['down']
    alarms=np.flatnonzero(valid & (scores>=threshold))
    first=int(alarms[0]) if len(alarms) else None
    lead=None if not label or first is None else round(float((shot['down']-times[first])*1000.),9)
    eligible=valid.copy()
    if label:eligible &= np.round((shot['down']-times)*1000.,9)>30.
    indices=np.flatnonzero(eligible)
    query=int(indices[np.argmax(scores[indices])]) if len(indices) else None
    return dict(shot=shot['shot'],label=label,query=query,
        probability=None if query is None else float(scores[query]),first_alarm_index=first,
        lead_ms=lead,timely=bool(lead is not None and lead>30.),
        false_alarm=bool(not label and first is not None),
        early_over_3000=bool(lead is not None and lead>3000.))


def fullcurve_guard(history, before, after, threshold):
    if (len({s['shot'] for s in history})!=len(history)
            or set(before)!=set(after) or set(before)!={s['shot'] for s in history}):
        raise ValueError('exact distinct ended trajectories required')
    old=[operational_state(s,before[s['shot']],threshold) for s in history]
    new=[operational_state(s,after[s['shot']],threshold) for s in history]
    lost_tp=[a['shot'] for a,b in zip(old,new) if a['timely'] and not b['timely']]
    lost_tn=[a['shot'] for a,b in zip(old,new)
             if not a['label'] and not a['false_alarm'] and b['false_alarm']]
    return dict(passed=not(lost_tp or lost_tn),lost_tp=lost_tp,lost_tn=lost_tn,
        checked_shots=len(history),before=old,after=new,
        scope='full actual causal intra trajectories, bounded ended history only')


def task_gradient(model, pair, *, threshold, intra):
    boundary=math.log(threshold/(1-threshold))
    gradient=torch.zeros_like(model.memory)
    records=[]
    for shot,query in pair:
        if intra:
            logit,info=causal_peak_logit(model,shot,query,create_graph=True,unroll_depth=4)
        else:
            x,a=_batch_windows(simple_data,{k:shot[k] for k in ('values','available')},
                               [query],model,model.center.device)
            logit=model(x,a,model.initial_fast(1))['logit'].reshape(())
            info=dict(mature_events=0,accepted_updates=0,differentiated_updates=0,fast_norm=0.)
        risk=F.softplus((1-2*int(shot['is_disrupt']))*(logit-boundary))
        g,=torch.autograd.grad(risk,model.memory)
        if not torch.isfinite(g).all():raise FloatingPointError('nonfinite operational memory gradient')
        gradient+=g.detach()/len(pair)
        records.append(dict(**info,shot=shot['shot'],label=int(shot['is_disrupt']),
            query=query,probability=float(logit.detach().sigmoid()),risk_loss=float(risk.detach())))
    return gradient,records


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _input_digest(shot):
    digest=hashlib.sha256()
    for key in ('values','available','times','valid'):
        value=np.ascontiguousarray(shot[key]);digest.update(key.encode())
        digest.update(str(value.shape).encode());digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    digest.update(repr((shot['is_disrupt'],shot['down'],shot['horizon'])).encode())
    return digest.hexdigest()


def _loss(state, threshold):
    # Saturated stored float32 probabilities cannot recover an unbounded logit.
    # This clamp is ONLY a numerical acceptance surrogate, never an emitted
    # score or alarm change. Actual training uses the original unbounded logit.
    p=np.clip(state['probability'],1e-7,1-1e-7)
    margin=math.log(float(p)/(1-float(p)))-math.log(threshold/(1-threshold))
    return float(np.logaddexp(0.,(1-2*state['label'])*margin))


class OperationalMemory:
    def __init__(self, model, *, bank, intra, step=.03, radius=1., source_weight=.20):
        if (not isinstance(intra,bool) or not all(math.isfinite(x) and x>0
                                                for x in (step,radius,source_weight))):
            raise ValueError('declared intra path and positive bounded update settings required')
        self.bank,self.intra,self.step,self.radius,self.source_weight=bank,intra,step,radius,source_weight
        self.rng=np.random.default_rng(271)
        self.last_index=0
        self.cache={}
        self.locked_head=None
        self.locked_threshold=None
        self.memory_sha=_digest(model.memory)
        model.requires_grad_(False)

    def _predict(self,model,shot):
        return predict_shot(model,shot,intra=self.intra,device=model.center.device)[0]

    def _source_guard(self,model,threshold):
        return evaluate_source_guard(model,self.bank,model.center.device,threshold)

    def _proposal(self,model,pair,threshold):
        model.requires_grad_(False);model.memory.requires_grad_(True)
        gradient,records=task_gradient(model,pair,threshold=threshold,intra=self.intra)
        loss,source=source_loss(model,self.bank,model.center.device,self.rng,
            batch_size=len(self.bank['labels']),indices=torch.arange(len(self.bank['labels'])))
        source_gradient,=torch.autograd.grad(loss,model.memory)
        gradient+=self.source_weight*source_gradient.detach()
        if not torch.isfinite(gradient).all():raise FloatingPointError('nonfinite combined memory gradient')
        proposal=normalized_step(model.memory.detach()[None],torch.zeros_like(model.memory)[None],
                                 gradient[None],step=self.step,radius=self.radius)[0]
        with torch.no_grad():model.memory.copy_(proposal)
        model.requires_grad_(False)
        return dict(gradient_norm=float(gradient.double().norm()),task_records=records,source=source)

    def __call__(self,model,shot,replay,*,provider,threshold,**kwargs):
        history=[*replay,shot]
        ids=[s['shot'] for s in history]
        index=len(provider.ended_ids)
        if (provider.pending is not None or not index or index<=self.last_index
                or len(set(ids))!=len(ids) or ids!=provider.ended_ids[-len(ids):]
                or any(s.get('device')!='EAST' or '_sealed_scores' not in s
                       or s.get('horizon')!=125. for s in history)):
            raise ValueError('new exact distinct sealed ended history suffix required')
        if not math.isfinite(threshold) or not 0<threshold<1:
            raise ValueError('finite fixed threshold required')
        if index<=30:
            if model.memory.any():raise AssertionError('no persistent FFN learning before coordinate lock')
            self.last_index=index
            return dict(accepted=False,reason='first30_auxiliary_learning_only',actual_steps=0,
                        effective_from_index=index+1)
        current_head=head_digest(model)
        if ((self.locked_head is not None and current_head!=self.locked_head)
                or (self.locked_threshold is not None and threshold!=self.locked_threshold)
                or _digest(model.memory)!=self.memory_sha):
            raise ValueError('external head, alarm or memory change invalidates replay cache')
        before_memory=model.memory.detach().clone()
        saved_cache=self.cache.copy()
        saved_rng=deepcopy(self.rng.bit_generator.state)
        accepted=False
        receipt=dict(accepted=False,actual_steps=0,effective_from_index=index+1,
            intra_in_task_and_guard=self.intra,unroll_depth=4,full_guard_replay=True,
            source_guard_scope='128 original J-TEXT TRAIN bank points, not whole source shots',
            memory_before_sha256=self.memory_sha)
        try:
            history=[provider.reencode(s) for s in history[-32:]]
            encoded={s['shot']:_input_digest(s) for s in history}
            before={}
            for s in history:
                cached=self.cache.get(s['shot'])
                if cached is not None and cached['input_sha256']!=encoded[s['shot']]:
                    raise ValueError('input coordinates changed after replay cache lock')
                before[s['shot']]=self._predict(model,s) if cached is None else cached['scores']
            states={s['shot']:operational_state(s,before[s['shot']],threshold) for s in history}
            current=history[-1]
            opposite=[s for s in history if int(s['is_disrupt'])!=int(current['is_disrupt'])
                      and states[s['shot']]['query'] is not None]
            if states[current['shot']]['query'] is None or not opposite:
                receipt['reason']='no_eligible_balanced_ended_operational_pair'
            else:
                pair_shots=[current,opposite[int(self.rng.integers(len(opposite)))]]
                pair_shots.sort(key=lambda s:int(s['is_disrupt']))
                pair=[(s,states[s['shot']]['query']) for s in pair_shots]
                proposal=self._proposal(model,pair,threshold)
                after={s['shot']:self._predict(model,s) for s in history}
                guard=fullcurve_guard(history,before,after,threshold)
                source=self._source_guard(model,threshold)
                before_loss=float(np.mean([_loss(states[s['shot']],threshold) for s in pair_shots]))
                new_states={s['shot']:operational_state(s,after[s['shot']],threshold) for s in pair_shots}
                after_loss=float(np.mean([_loss(new_states[s['shot']],threshold) for s in pair_shots]))
                tolerance=float(8*np.finfo(np.float32).eps*max(abs(before_loss),abs(after_loss),1e-12))
                accepted=guard['passed'] and source['passed'] and before_loss-after_loss>tolerance
                receipt.update(actual_steps=1,proposal=proposal,guard=guard,source_guard=source,
                    pair_ids=[s['shot'] for s in pair_shots],task_before=before_loss,
                    task_after=after_loss,task_gain=before_loss-after_loss,task_tolerance=tolerance,
                    reason='accepted_full_operational_replay' if accepted else
                           'history_crossing' if not guard['passed'] else
                           'source_point_crossing' if not source['passed'] else 'task_not_improved')
                if accepted:before=after
            if not accepted:
                with torch.no_grad():model.memory.copy_(before_memory)
            self.cache={s['shot']:dict(input_sha256=encoded[s['shot']],scores=before[s['shot']].copy())
                        for s in history}
        except Exception:
            with torch.no_grad():model.memory.copy_(before_memory)
            self.cache=saved_cache;self.rng.bit_generator.state=saved_rng
            raise
        finally:
            model.requires_grad_(False);model.zero_grad(set_to_none=True)
        self.last_index=index
        self.locked_head=current_head;self.locked_threshold=threshold
        self.memory_sha=_digest(model.memory)
        receipt.update(accepted=bool(accepted),history_ids=[s['shot'] for s in history],
            memory_after_sha256=self.memory_sha,
            memory_change_norm=float((model.memory.detach()-before_memory).double().norm()),
            memory_norm=float(model.memory.detach().double().norm()))
        return receipt
