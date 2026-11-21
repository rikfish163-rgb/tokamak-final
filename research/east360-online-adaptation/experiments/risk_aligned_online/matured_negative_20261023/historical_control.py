"""Matched historical-only retraining: current signals are prediction inputs only."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..feedback_gradient_20261023.geometry import lead_ms,trial_step
from ..heldout_threshold_20261023.run import models_for_calibration
from ..history_tangent_20261023.learning import logit
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import tensors
from .run import REFERENCE,historical_replay

PRIMARY=io23.ROOT/'experiments/risk_aligned_online/runs/matured_negative360_factor_matched_20261023'
PRIMARY_SHA='2d04ba51abc6b84064a32d0af0a0a6c570fbe2fa914656e4dc73da32dc33ffbb'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/已度过窗口反馈_历史再训练对照约定_20261023.md'


def fixed_objective(negative_logits,positive_logits):
    return .5*(F.softplus(negative_logits).mean()+F.softplus(-positive_logits).mean())


def historical_step(model,fast,negative,positive):
    def objective(state):
        return fixed_objective(logit(model,*negative,state),logit(model,*positive,state))
    loss=objective(fast)
    gradients=torch.autograd.grad(loss,tuple(fast.values()))
    norm=sum(g.square().sum() for g in gradients).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite historical update')
    new=trial_step(fast,gradients)
    distance=sum((new[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt()
    with torch.no_grad():after=objective(new)
    return new,dict(loss_before=float(loss.detach()),loss_after=float(after),
                    gradient_norm=float(norm),parameter_delta_norm=float(distance))


def state_index(prediction_index):
    if prediction_index<0:raise ValueError('nonnegative prediction index required')
    return max(0,prediction_index//128-1)


def negative_replay(positive_manifest,mapping):
    ids=sorted({r['shot'] for r in positive_manifest});windows=[];manifest=[]
    reader=io23.read_outcome
    def guard(device,sid):
        assert device=='EAST' and sid in ids
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guard):
        for sid in ids:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            assert s['is_disrupt']
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            leads=lead_ms(s['times'],s['down'])
            eligible=np.flatnonzero(s['valid'] & (s['y']==0) & (leads>125) & (np.arange(len(leads))>=127))
            if not len(eligible):raise ValueError(f'no complete historical negative: {sid}')
            for desired in (500.,250.):
                end=int(eligible[np.argmin(abs(leads[eligible]-desired))])
                windows.append(simple_data.window(signal,end,128))
                manifest.append(dict(shot=sid,index=end,desired_lead_ms=desired,actual_lead_ms=float(leads[end])))
    assert len(windows)==8
    return tuple(torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)),manifest


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    assert sha(PRIMARY/'completed.json')==PRIMARY_SHA
    done=READ(PRIMARY/'completed.json');protocol=READ(PRIMARY/'protocol.json')
    for name,digest in done['artifact_sha256'].items():assert sha(PRIMARY/name)==digest,name
    for name,digest in protocol['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    for name,digest in protocol['input_sha256'].items():assert sha(name)==digest,name
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];initial=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json')
    positive,pm=historical_replay(protocol['preparation_ids'],mapping)
    assert pm==protocol['replay_manifest']
    negative,nm=negative_replay(pm,mapping)
    root.mkdir(parents=True);(root/'shots').mkdir()
    files={Path(__file__),io23.ROOT/'experiments/risk_aligned_online/tests/test_matured_historical.py'}
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(primary_sha256=PRIMARY_SHA,plan_sha256=sha(PLAN),negative_manifest=nm,positive_manifest=pm,
           shot_ids=protocol['shot_ids'],threshold=protocol['threshold'],source_threshold=protocol['source_threshold'],
           code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},source_sha256=PARENT_SHA256,
           loss_coefficients=dict(negative=.5,replay=.5),learning_rate=.001,clip_norm=1.,
           scope='POST_PRIMARY_DIAGNOSTIC: historical-only updates; no current signal feedback',
           additional_target_label_identities=0,inter_updates=0,alarm_updates=0,independent_confirmation=False)
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    states=[initial_encoder(model)];updates=[];receipts=[]
    def deny(*args,**kwargs):raise AssertionError('no current/future outcome access during prediction')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in p['shot_ids']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=dict(transform({k:s[k] for k in ('values','available')},mapping),times=s['times'])
            scores=[]
            for start in range(0,len(s['times']),32):
                index=state_index(start)
                while len(states)<=index:
                    fast,r=historical_step(model,states[-1],negative,positive)
                    states.append(fast);updates.append(r)
                x,m=tensors(signal,range(start,min(start+32,len(s['times']))),'cuda')
                with torch.no_grad():scores.append(logit(model,x,m,states[index]).sigmoid().cpu().numpy())
            curve=np.concatenate(scores)
            if not np.isfinite(curve).all():raise FloatingPointError('nonfinite control curve')
            with np.load(PRIMARY/'shots'/f'{sid}.npz') as z:np.testing.assert_array_equal(curve[:256],z['static'][:256])
            path=root/'shots'/f'{sid}.npz';np.savez_compressed(path,times=s['times'],scores=curve)
            receipts.append(dict(shot=sid,prediction_sha256=sha(path),logical_updates=state_index(len(curve)-1)))
            if len(receipts)%60==0:print(dict(shots=len(receipts),seconds=time.monotonic()-started),flush=True)
    _durable_json(root/'prediction_receipts.json',receipts)
    _durable_json(root/'cached_update_sequence.json',updates)
    # Verify cached parameter reuse is exactly independent retraining, not a
    # persistent state carried across different shots.
    again=initial_encoder(model)
    for i in range(min(3,len(updates))):
        again,r=historical_step(model,again,negative,positive)
        assert r==updates[i]
        for k in again:torch.testing.assert_close(again[k],states[i+1][k],rtol=0,atol=0)
    records=[]
    for sid in p['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz') as z:
            np.testing.assert_array_equal(z['times'],s['times'])
            records.append(dict(shot=sid,times=s['times'][s['valid']],scores=z['scores'][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    results={}
    for level,tau in [('calibrated',p['threshold']),('source_threshold',p['source_threshold'])]:
        r=metrics(records,tau,lead=30.)
        r['very_early_over3000']=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions'])
        results[level]=r
    _durable_json(root/'per_shot.json',results)
    assert_unchanged(model,initial);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    final=dict(status='COMPLETED_HISTORICAL_CONTROL',metrics={k:public(v) for k,v in results.items()},
               cached_gradient_steps=len(updates),logical_per_shot_updates=sum(r['logical_updates'] for r in receipts),
               seconds=time.monotonic()-started,source_unchanged=True,independent_confirmation=False,
               artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',final)
    print({k:v for k,v in final.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
