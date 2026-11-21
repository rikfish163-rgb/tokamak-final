"""Recompute selected training objectives; real-model chain rule and stream replay."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..feedback_cone_feasibility_20261024.stream import signal_parts
from ..history_tangent_20261023.learning import logit,flat,apply_step
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import (ARMS,feedback_context,load_policy,evaluate_policy,predict,eligible,
                       policy_step,outer_gradient)
from .run import PREVIOUS


def main(root):
    done=READ(root/'completed.json');assert done['status']=='completed'
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    p=READ(root/'protocol.json');records=READ(root/'policies.json');logs={}
    for kind in ('global','context'):
        logs[kind]=[READ(root/'training'/f'{kind}_{i:02d}.json') for i in range(11)]
        best=min(logs[kind],key=lambda r:r['objective'])
        assert best['policy']==records[kind]
        assert READ(root/f'{kind}_selection.json')['epoch']==best['epoch']
        for record in logs[kind]:
            total=0
            for group in ('safe','early','danger'):
                rs=[r for r in record['per_shot'] if r['group']==group]
                total+=sum(r['delta']+max(0,r['delta']) for r in rs)/(3*len(rs))
            np.testing.assert_allclose(total,record['objective'],atol=1e-12,rtol=1e-10)
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');cache=[];shots={};gradient_check=None
    receipt_lookup={(r['shot'],r['count']):r for r in READ(root/'cache_receipts.json')}
    def deny(*args,**kwargs):raise AssertionError('no additional outcomes during audit')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in p['preparation_ids']:
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            old=READ(PREVIOUS/'geometry'/f'{sid}.json');rows=old['rows']
            shots[sid]=dict(signal=s,times=raw['times'],flag=old['is_disrupt'],down=old['down'])
            for count in sorted({r['count'] for r in rows}):
                fast=initial_encoder(model);parts,_=signal_parts(model,s,count,fast,hold=False,device='cuda')
                c,observed=feedback_context(model,s,count,fast,hold=False,device='cuda')
                scale=float(parts.double().norm(dim=1).max());receipt=receipt_lookup[(sid,count)]
                assert scale==receipt['scale'];np.testing.assert_array_equal(c.cpu().numpy(),receipt['context'])
                np.testing.assert_array_equal(observed.cpu().numpy(),receipt['observed'])
                np.testing.assert_array_equal((parts.double()@parts.double().T).cpu().numpy(),receipt['channel_gram'])
                cache.append(dict(shot=sid,count=count,parts=parts.cpu(),scale=scale,context=c.cpu(),observed=observed.cpu(),
                                  rows=[r for r in rows if r['count']==count]))
                if sid==63226 and count==3712:
                    policy=load_policy(records['context'],device='cuda');step=policy_step(parts,policy(c,observed))
                    # End-to-end reference does NOT detach the inner parameter
                    # change. It differentiates the real Transformer to policy.
                    ref={};offset=0
                    for k,v in fast.items():
                        n=v.numel();ref[k]=v.detach()-step[offset:offset+n].reshape_as(v);offset+=n
                    x,m=[torch.as_tensor(v[None],device='cuda') for v in simple_data.window(s,3817,128)]
                    z=logit(model,x,m,ref);loss=torch.nn.functional.softplus(z).mean()
                    reference=flat(torch.autograd.grad(loss,tuple(policy.parameters()))).detach()
                    changed=apply_step(fast,step.detach());z=logit(model,x,m,changed)
                    r=flat(torch.autograd.grad(torch.nn.functional.softplus(z).mean(),tuple(changed.values()))).detach()
                    analytic=outer_gradient(policy,c,observed,parts,scale,r)
                    torch.testing.assert_close(analytic,reference,atol=1e-9,rtol=2e-5)
                    gradient_check=dict(shot=sid,count=count,end=3817,maximum_absolute_error=float((analytic-reference).abs().max()),
                                        reference_norm=float(reference.norm()))
            print(dict(stage='audit_cache',shot=sid,blocks=len(cache)),flush=True)
        objective_replays={}
        for kind in ('global','context'):
            loss,gradient,details=evaluate_policy(model,load_policy(records[kind],device='cuda'),cache,shots)
            best=min(logs[kind],key=lambda r:r['objective'])
            assert loss==best['objective'] and details==best['per_shot']
            assert float(np.linalg.norm(gradient))==best['gradient_norm']
            objective_replays[kind]=dict(epoch=best['epoch'],objective=loss)
        del cache,shots
        replays=[];linear=READ(PREVIOUS/'policy.json')['block']
        for sid in (89272,89284,89584):
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,s,records,linear)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for arm in ('off',)+ARMS:np.testing.assert_array_equal(curves[arm],z[arm])
            assert updates==READ(root/'updates'/f'{sid}.json');replays.append(sid)
    outcomes={r['shot']:r for r in READ(root/'outcomes.json')};counts={};independent={}
    for arm in ('off',)+ARMS:
        tp=fp=early=0;danger=[];safe=[]
        for sid in p['shot_ids']:
            o=outcomes[sid]
            with np.load(root/'shots'/f'{sid}.npz') as z:
                lead=np.round((o['down']-z['times'])*1000,9) if o['is_disrupt'] else None
                valid=lead>=-1e-6 if o['is_disrupt'] else np.ones(len(z['times']),bool)
                scores=z[arm][valid].astype(np.float64);alarms=np.flatnonzero(scores>=p['threshold'])
                if o['is_disrupt']:
                    timely=bool(len(alarms) and lead[valid][alarms[0]]>30)
                    tp+=int(timely);early+=int(timely and lead[valid][alarms[0]]>3000)
                    danger.append(float(-np.log(scores[(lead[valid]>30)&(lead[valid]<=125)].clip(1e-12,1)).mean()))
                else:
                    fp+=int(bool(len(alarms)));safe.append(float(-np.log1p(-scores.max().clip(0,1-1e-12))))
        independent[arm]=dict(tp=tp,fp=fp,very_early_over3000=early,danger_loss=float(np.mean(danger)),
                              safe_loss=float(np.mean(safe)),balanced_loss=float((np.mean(danger)+np.mean(safe))/2))
        for k,v in independent[arm].items():np.testing.assert_allclose(v,done['results'][arm][k],atol=1e-12,rtol=1e-10)
    for sid in p['shot_ids']:
        updates=READ(root/'updates'/f'{sid}.json')
        for arm,rs in updates.items():
            counts[arm]=counts.get(arm,0)+sum(r['gradient_steps'] for r in rs)
            for i,r in enumerate(rs):
                assert r['first_affected_prediction']==r['last_observation']+1==256+128*i
                assert r['actual_step_norm']<=.01001
                if arm in ('uniform','permuted','hold') and not r['control_zero_direction']:
                    np.testing.assert_allclose(r['requested_step_norm'],r['main_requested_norm'],atol=1e-9,rtol=1e-6)
    assert eligible(done['results'])==done['eligible_for360']
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='passed',completed_sha256=sha(root/'completed.json'),training_objective_recounts=22,
        cached_blocks_rebuilt=len(receipt_lookup),selected_objectives_replayed=objective_replays,real_model_chain_rule=gradient_check,
        complete_shots_replayed=replays,independent_metrics=independent,gradient_updates=counts,
        source_and_parent_unchanged=True,audit_code_sha256=sha(Path(__file__)))
    _durable_json(root/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
