"""Parent-reset internal encoder updates driven only by arrived signals."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..history_tangent_20261023.learning import flat, logit, matched_step, apply_step
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..signal_driven_span_20261024.run import evaluate
from ..warmstart_alignment_20261022.run import predict_many
from .policy import decide
from .run import (READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,signal,transform,
                  REFERENCE,DEFAULT_PARENT,PARENT_SHA256,CALIBRATION_IDS)

ARMS=('mean001','mean010','uniform001','uniform010','hold001','hold010')
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/平均通道收益完整炮流_事前约定_20261024.md'


def signal_parts(model,signal,count,fast,*,hold,device):
    if set(signal)!={'values','available'} or type(count) is not int or not 256<=count<=len(signal['values']):
        raise ValueError('signal-only arrived prefix and four matured targets required')
    prefix={k:v[:count] for k,v in signal.items()}
    x,m,truth,tm,origins=forecast_examples(prefix,count,model.cfg,device)
    valid=target_mask(x,m,truth,tm)
    receipt=dict(last_observation=count-1,first_affected_prediction=count,origins=origins,observed=int(valid.sum()),hold=hold)
    if not valid.any():
        receipt['reason']='no_measured_targets'
        return None,receipt
    target=x[:,-1:].expand_as(truth) if hold else truth
    pred=model.predict_fast(x,m,fast)['forecast']
    parts=[]
    for c in range(23):
        cm=torch.zeros_like(valid);cm[:,:,c]=valid[:,:,c]
        loss=model.base.forecast_loss(pred,target,cm)*cm.sum()/valid.sum()
        parts.append(flat(torch.autograd.grad(loss,tuple(fast.values()),retain_graph=True)).detach())
    parts=torch.stack(parts)
    if not torch.isfinite(parts).all():raise FloatingPointError('nonfinite arrived gradient')
    return parts,receipt


def predict(model,signal,prior,*,device='cuda'):
    prior=np.asarray(prior,dtype=np.float64)
    if set(signal)!={'values','available'} or prior.shape!=(23,) or not np.isfinite(prior).all():
        raise ValueError('signal-only input and fixed23 historical utility required')
    states={a:initial_encoder(model) for a in ('off',)+ARMS};curves={a:[] for a in states};receipts={a:[] for a in ARMS}
    n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(n,lo+128)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,end,128) for end in range(start,min(hi,start+32))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                for arm in states:curves[arm].append(logit(model,x,m,states[arm]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        fast=initial_encoder(model)
        real,rr=signal_parts(model,signal,hi,fast,hold=False,device=device)
        fake,fr=signal_parts(model,signal,hi,fast,hold=True,device=device)
        for arm in ARMS:
            parts,r=(fake,fr) if arm.startswith('hold') else (real,rr)
            r=dict(r);length=.001 if arm.endswith('001') else .01
            if parts is None:
                direction=torch.zeros(sum(v.numel() for v in fast.values()),device=device)
                weights=np.zeros(23);decision=dict(abstain=True,reason='no_measured_targets')
            else:
                g=parts.sum(0);norm=float(g.double().norm())
                if arm.startswith('uniform'):
                    weights=np.ones(23)/23;decision=dict(abstain=norm==0)
                elif norm==0:
                    weights=np.zeros(23);decision=dict(abstain=True,reason='zero_signal_gradient')
                else:
                    b=(parts.double()@g.double()/g.double().square().sum()).cpu().numpy()
                    weights,decision=decide(prior,b)
                direction=torch.as_tensor(weights,dtype=parts.dtype,device=device)@parts
            step=matched_step(direction,length);states[arm]=apply_step(fast,step)
            actual=flat([states[arm][k].detach()-v.detach() for k,v in fast.items()])
            r.update(weights=weights.tolist(),decision=decision,requested_norm=length,
                     actual_step_norm=float(actual.double().norm()),gradient_steps=int(bool(step.any())),reset='parent')
            receipts[arm].append(r)
    return {a:np.concatenate(v) for a,v in curves.items()},receipts


def gate(results):
    off=results['off']
    for suffix in ('001','010'):
        c=results['mean'+suffix]
        if not (c['tp']>=off['tp'] and c['fp']<=off['fp'] and c['danger_loss']<=off['danger_loss']
                and c['safe_loss']<=off['safe_loss']
                and all(c['balanced_loss']<results[a+suffix]['balanced_loss']-1e-6 for a in ('uniform','hold'))):
            return False
    return True


def main(policy_root,root):
    if root.exists():raise FileExistsError(root)
    done=READ(policy_root/'completed.json');assert done['status']=='completed'
    for f,d in done['artifact_sha256'].items():assert sha(policy_root/f)==d
    prior=np.array(READ(policy_root/'policies.json')['None']['context']['target_mean'])
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed']
    original=state_cpu(model);mapping=READ(REFERENCE/'input_map.json')
    root.mkdir(parents=True);(root/'shots').mkdir();(root/'updates').mkdir()
    p=dict(shot_ids=CALIBRATION_IDS,threshold=.12540414929389954,arms=('off',)+ARMS,
           prior=prior.tolist(),policy_sha256=sha(policy_root/'policies.json'),plan_sha256=sha(PLAN),
           source_sha256=PARENT_SHA256,parent_sha256=sha(REFERENCE/'fold0/mapped_fixed/prepared.pt'),
           mapping_sha256=sha(REFERENCE/'input_map.json'),update_every=128,first_update=256,
           fast_reset='parent each update',step_norms=[.001,.01],current_label_access='only after all predictions sealed',
           scope='DEVELOPMENT10_AFTER_PROBE_EXPLORATION_NOT_INDEPENDENT',inter_steps=0,alarm_updates=0,
           code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')},
           test_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_mean_channel_stream.py'))
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in Path(__file__).parent.glob('*.py'):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('no current outcomes before complete prediction seal')
    seals=[]
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in CALIBRATION_IDS:
            raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,mapped,prior)
            with torch.no_grad():baseline=predict_many(model,dict(parent=mapped),device='cuda')['parent']
            np.testing.assert_array_equal(curves['off'],baseline)
            for arm in ARMS:np.testing.assert_array_equal(curves[arm][:256],baseline[:256])
            path=root/'shots'/f'{sid}.npz';np.savez_compressed(path,times=raw['times'],**curves)
            up=root/'updates'/f'{sid}.json';_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
            print(dict(stage='prediction',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True)
            if time.monotonic()-start>900:raise TimeoutError('15minute declared screen exceeded')
    _durable_json(root/'prediction_seals.json',seals)
    # Reuse established metric implementation, replacing its arm constant only
    # in this explicit call; same strict lead and shot-level loss definitions.
    from ..signal_driven_span_20261024 import run as prior_run
    with patch.object(prior_run,'ARMS',ARMS):results=evaluate(root,p)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='completed',seconds=time.monotonic()-start,results=results,eligible_for360=gate(results),
                source_and_parent_unchanged=True,artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',result);print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.policy,args.out)
