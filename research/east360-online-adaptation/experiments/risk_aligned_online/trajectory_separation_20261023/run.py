"""Replay five declared shots, quantify amplification, then conditional probes.

Outcome metadata selects diagnostic endpoints only. It never reaches an update;
all temporary trajectories are verified against the previous sealed receipts.
"""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..channel_gradient_audit import channel_losses
from ..credit_horizon_20261023.run import READ
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..external65_20261023.run import models_for,MODELS
from ..first_feedback_20261023.learning import schedule,update_latest
from ..history_tangent_20261023.learning import flat,logit
from ..increment_forecast_20261023.learning import update,target_mask
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .analysis import separation,parameter_group,channel_separation

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/first_feedback65_20261023'
REF_SHA='a80ad4a52c7c9012813005fd40aa559f20082a8bbf173bf08bccea7087abb853'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/炮内轨迹分离_梯度诊断约定_20261023.md'
IDS=(74412,75604,77412,77639,88785)
ARMS=('early1','late1')


def copy_state(state):
    return {k:v.detach().clone().requires_grad_(True) for k,v in state.items()}


def gradients(model,prefix,count,fast,*,components=False):
    x,m,truth,tm,_=forecast_examples(prefix,count,model.cfg,'cuda')
    mask=target_mask(x,m,truth,tm)
    forecast=model.predict_fast(x,m,fast)['forecast']
    loss=model.base.forecast_loss(forecast,truth,mask)
    grad=flat(torch.autograd.grad(loss,tuple(fast.values()),retain_graph=components)).detach()
    if not components:return grad,float(loss.detach())
    parts=channel_losses(forecast,truth,mask,model.base.center,model.base.scale)
    torch.testing.assert_close(parts.sum(),loss)
    by_channel=torch.stack([flat(torch.autograd.grad(parts[c],tuple(fast.values()),retain_graph=c<22)).detach()
                            for c in range(23)])
    torch.testing.assert_close(by_channel.sum(0),grad,rtol=3e-4,atol=2e-6)
    return grad,float(loss.detach()),by_channel,parts.detach(),mask.sum((0,1))


def inspect(model,signal,critical,old_receipts):
    n=len(signal['values']);states={a:initial_encoder(model) for a in ARMS}
    schedules={a:set(schedule(n,a)) for a in ARMS};steps=[];receipts={a:[] for a in ARMS}
    best=None;best_states=None;critical_states=None
    for count in sorted(set.union(*schedules.values())):
        if critical<count and critical_states is None:
            critical_states={a:copy_state(s) for a,s in states.items()}
        prefix={k:v[:count] for k,v in signal.items()}
        shared=all(count in s for s in schedules.values())
        if shared:
            gs={a:gradients(model,prefix,count,states[a]) for a in ARMS}
            row=dict(count=count,loss_early=gs['early1'][1],loss_late=gs['late1'][1],
                **separation(flat(states['early1'].values()),flat(states['late1'].values()),
                             gs['early1'][0],gs['late1'][0]))
            if row['distance_ratio'] is not None and (best is None or row['distance_ratio']>best['distance_ratio']):
                best=row;best_states={a:copy_state(s) for a,s in states.items()}
        for a in ARMS:
            if count in schedules[a]:
                fn=update_latest if count<384 else update
                states[a],r=fn(model,prefix,count,states[a],device='cuda');receipts[a].append(r)
        if shared:
            actual=float((flat(states['early1'].values()).detach().double()-flat(states['late1'].values()).detach().double()).norm())
            row['actual_distance_after']=actual
            # Actual float32 SGD rounding is recorded, not hidden in the algebraic double-precision identity.
            row['float32_distance_error']=abs(actual-row['distance_after_squared']**.5)
            steps.append(row)
    if critical_states is None:critical_states={a:copy_state(s) for a,s in states.items()}
    assert receipts=={a:old_receipts[a] for a in ARMS}
    if best is None:raise ValueError('no common update for diagnostic')
    prefix={k:v[:best['count']] for k,v in signal.items()}
    ga,la,ca,pa,na=gradients(model,prefix,best['count'],best_states['early1'],components=True)
    gb,lb,cb,pb,nb=gradients(model,prefix,best['count'],best_states['late1'],components=True)
    torch.testing.assert_close(na,nb,rtol=0,atol=0)
    d=flat(best_states['early1'].values()).detach()-flat(best_states['late1'].values()).detach()
    components=channel_separation(d,ca,cb,best['scale_early'],best['scale_late'],.001)
    for c,r in enumerate(components):
        r.update(channel=io23.RAW23[c],observations=int(na[c]),loss_early=float(pa[c]),loss_late=float(pb[c]))
    actual_linear=best['linear_distance_squared_change']
    decomposition=sum(r['linear_distance_squared_change'] for r in components)
    np.testing.assert_allclose(decomposition,actual_linear,atol=1e-10,rtol=1e-3)
    # Use the original batch boundaries, preserving numerical dispatch.
    bounds=sorted({0,n,*range(32,n,32),*(c for s in schedules.values() for c in s)})
    lo=max(v for v in bounds if v<=critical);hi=min(v for v in bounds if v>critical)
    windows=[simple_data.window(signal,e,128) for e in range(lo,hi)]
    x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
    def probability(state):
        with torch.no_grad():return float(logit(model,x,m,state).sigmoid()[critical-lo])
    endpoint={a:probability(s) for a,s in critical_states.items()}
    groups=sorted({parameter_group(k) for k in critical_states['early1']})
    substitutions={}
    for group in groups:
        mixed={k:(critical_states['early1'][k] if parameter_group(k)==group else v)
               for k,v in critical_states['late1'].items()}
        substitutions[group]=probability(mixed)
    return dict(critical_index=critical,endpoint_probabilities=endpoint,
        substitutions_into_late=substitutions,steps=steps,maximum_amplification=best,
        channel_components=components,linear_decomposition_error=abs(decomposition-actual_linear),
        receipts_exact=True,updates_each=len(receipts['early1']))


def main(root):
    if root.exists():raise FileExistsError(root)
    assert sha(REFERENCE/'completed.json')==REF_SHA
    prior=READ(REFERENCE/'protocol.json');done=READ(REFERENCE/'completed.json')
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert not set(IDS)&set(io23.cohort_ids('EAST'))
    for path,h in prior['checkpoints'].items():assert sha(path)==h
    for sid in IDS:assert sha(io23.path_for('EAST',sid))==prior['input_sha256'][str(io23.path_for('EAST',sid))]
    root.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(131)
    codes=[Path(__file__),Path(__file__).with_name('analysis.py'),
        io23.ROOT/'experiments/risk_aligned_online/tests/test_trajectory_separation.py']
    _durable_json(root/'protocol.json',dict(reference_sha256=REF_SHA,shots=IDS,plan_sha256=sha(PLAN),
        code_sha256={str(p):sha(p) for p in codes},checkpoints=prior['checkpoints'],
        interpretation='POSTHOC_DIAGNOSTIC',test360_accessed=False,policy_changed=False))
    roster={r['shot']:r for r in prior['roster']};results=[]
    def deny(*args,**kwargs):raise AssertionError('no outcome reading in neural diagnostic')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for f in range(4):
            models=models_for(f);frozen={m:state_cpu(v) for m,v in models.items()}
            mapping=READ(REFERENCE/f'fold{f}'/'input_map.json')
            for sid in IDS:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                path=REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz'
                receipts_path=path.with_name(f'{sid}_updates.json')
                for pp in (path,receipts_path):assert sha(pp)==done['artifact_sha256'][str(pp.relative_to(REFERENCE))]
                with np.load(path) as curves:
                    valid=np.ones(len(s['times']),bool) if not roster[sid]['is_disrupt'] else (
                        roster[sid]['down']-s['times'])*1000>=-1e-6
                    for name,model in models.items():
                        early,late=[curves[f'{name}_{a}'] for a in ARMS]
                        critical=int(np.argmax(np.where(valid,np.abs(early-late),-1)))
                        result=inspect(model,signal,critical,READ(receipts_path)[name])
                        for a in ARMS:np.testing.assert_allclose(result['endpoint_probabilities'][a],curves[f'{name}_{a}'][critical],rtol=0,atol=2e-6)
                        result.update(fold=f,shot=sid,model=name,is_disrupt=roster[sid]['is_disrupt'],
                            critical_time=float(s['times'][critical]),
                            prior_max_difference=float(abs(early[critical]-late[critical])))
                        _durable_json(root/f'fold{f}_{sid}_{name}.json',result);results.append(result)
                        print(dict(fold=f,shot=sid,model=name,max_ratio=result['maximum_amplification']['distance_ratio'],
                            endpoint_difference=result['prior_max_difference'],seconds=round(time.monotonic()-start,2)),flush=True)
            for name,model in models.items():assert_unchanged(model,frozen[name])
    for path,h in prior['checkpoints'].items():assert sha(path)==h
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    summary=dict(records=len(results),distinct_shots=len(IDS),
        common_updates=sum(len(r['steps']) for r in results),
        amplifying_updates=sum(s['distance_ratio']>1 for r in results for s in r['steps'] if s['distance_ratio'] is not None),
        maximum_ratio=max(r['maximum_amplification']['distance_ratio'] for r in results),
        max_replay_probability_error_tolerance=2e-6,receipts_all_exact=True,
        source_unchanged=True,checkpoints_unchanged=True,test360_accessed=False,
        status='COMPLETED_POSTHOC_DIAGNOSTIC',seconds=time.monotonic()-start,
        artifact_sha256={str(p.relative_to(root)):sha(p) for p in root.glob('*') if p.is_file()})
    _durable_json(root/'completed.json',summary);print({k:v for k,v in summary.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
