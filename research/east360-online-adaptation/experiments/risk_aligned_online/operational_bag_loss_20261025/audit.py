"""Independent arithmetic, real-model finite chain rule and exact stream replays."""
import argparse
import gc
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from .. import io23,simple_data
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..finite_conditional_loss_20261025.learning import load_policy,policy_step,outer_gradient
from ..history_tangent_20261023.learning import logit,flat,apply_step
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ARMS,evaluate_policy,predict,eligible,bag_loss
from .run import PREVIOUS,build_history


def main(root):
    done=READ(root/'completed.json');assert done['status']=='completed'
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    p=READ(root/'protocol.json');records=READ(root/'policies.json');best={}
    for path,digest in p['input_sha256'].items():assert sha(Path(path))==digest,path
    for kind in ('global','context'):
        logs=[READ(root/'training'/f'{kind}_{i:02d}.json') for i in range(11)]
        best[kind]=min(logs,key=lambda r:r['objective'])
        assert best[kind]['policy']==records[kind]
        assert best[kind]['epoch']==READ(root/f'{kind}_selection.json')['epoch']
        for r in logs:
            total=0.
            for flag in (False,True):
                ds=[x['delta'] for x in r['per_shot'] if bool(x['flag'])==flag]
                total+=sum(d+max(d,0) for d in ds)/(2*len(ds))
            np.testing.assert_allclose(total,r['objective'],atol=1e-12,rtol=1e-10)
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');chain=None
    def deny(*args,**kwargs):raise AssertionError('new outcomes forbidden during audit')
    with patch.object(io23,'read_outcome',side_effect=deny):
        shots,receipts,_=build_history(model,p['preparation_ids'],mapping,p['threshold'])
        assert receipts==READ(root/'cache_receipts.json')
        assert {str(k):v['baseline'] for k,v in shots.items()}==READ(root/'historical_baselines.json')
        objective_replays={}
        for kind in ('global','context'):
            policy=load_policy(records[kind],device='cuda')
            loss,g,detail,gate=evaluate_policy(model,policy,shots,p['threshold'])
            r=best[kind]
            assert loss==r['objective'] and detail==r['per_shot'] and gate==r['history_gate']
            assert float(np.linalg.norm(g))==r['gradient_norm']
            objective_replays[kind]=dict(epoch=r['epoch'],objective=loss,history_gate=gate)
            print(dict(stage='selected_objective_replayed',kind=kind,objective=loss),flush=True)
        # Fixed first disruptive historical shot; use its selected-policy winner.
        row=next(r for r in best['context']['per_shot'] if r['shot']==63226)
        if row['count']<256:raise AssertionError('declared real-model derivative probe has no update')
        shot=shots[63226];item=shot['cache'][row['count']];fast=initial_encoder(model)
        parts=item['parts'].to('cuda');c=item['context'].to('cuda');observed=item['observed'].to('cuda')
        policy=load_policy(records['context'],device='cuda');step=policy_step(parts,policy(c,observed))
        ref={};offset=0
        for k,v in fast.items():
            n=v.numel();ref[k]=v.detach()-step[offset:offset+n].reshape_as(v);offset+=n
        windows=[simple_data.window(shot['signal'],int(e),128) for e in row['chunk']]
        x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
        z=logit(model,x,m,ref)[row['position']]
        reference=flat(torch.autograd.grad(bag_loss(z,True,p['threshold']),tuple(policy.parameters()))).detach()
        changed=apply_step(fast,step.detach());z=logit(model,x,m,changed)[row['position']]
        risk=flat(torch.autograd.grad(bag_loss(z,True,p['threshold']),tuple(changed.values()))).detach()
        analytic=outer_gradient(policy,c,observed,parts,item['scale'],risk)
        torch.testing.assert_close(analytic,reference,atol=1e-9,rtol=2e-5)
        chain=dict(shot=63226,count=row['count'],end=row['end'],
                   max_absolute_error=float((analytic-reference).abs().max()),reference_norm=float(reference.norm()))
        del shots;gc.collect()
        replays=[];stage_record=READ(PREVIOUS/'policies.json')['context']
        for sid in (89272,89284,89584):
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,s,records,stage_record)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for arm in ('off',)+ARMS:np.testing.assert_array_equal(curves[arm],z[arm])
            assert updates==READ(root/'updates'/f'{sid}.json');replays.append(sid)
    outcomes={r['shot']:r for r in READ(root/'outcomes.json')};independent={};updates_count={}
    b=math.log(p['threshold']/(1-p['threshold']))
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
                    peak=scores[lead[valid]>30].max()
                else:fp+=int(bool(len(alarms)));peak=scores.max()
                peak=np.clip(peak,1e-12,1-1e-12);zpeak=np.log(peak/(1-peak))
                target=danger if o['is_disrupt'] else safe
                target.append(float(np.logaddexp(0,b-zpeak if o['is_disrupt'] else zpeak-b)))
        independent[arm]=dict(tp=tp,fp=fp,very_early_over3000=early,bag_disrupt_loss=float(np.mean(danger)),
            bag_safe_loss=float(np.mean(safe)),bag_balanced_loss=float((np.mean(danger)+np.mean(safe))/2))
        for k,v in independent[arm].items():np.testing.assert_allclose(v,done['results'][arm][k],atol=1e-12,rtol=1e-10)
    for sid in p['shot_ids']:
        for arm,rs in READ(root/'updates'/f'{sid}.json').items():
            updates_count[arm]=updates_count.get(arm,0)+sum(r['gradient_steps'] for r in rs)
            for i,r in enumerate(rs):
                assert r['first_affected_prediction']==r['last_observation']+1==256+128*i
                assert r['actual_step_norm']<=.01001
                if arm in ('uniform','permuted','hold') and not r['control_zero_direction']:
                    np.testing.assert_allclose(r['requested_step_norm'],r['main_requested_norm'],atol=1e-9,rtol=1e-6)
    assert eligible(done['results'],best['context']['history_gate']['passed'])==done['eligible_for360']
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='passed',completed_sha256=sha(root/'completed.json'),training_objective_recounts=22,
        cached_blocks_rebuilt=len(receipts),selected_objectives_replayed=objective_replays,real_model_chain_rule=chain,
        complete_shots_replayed=replays,independent_metrics=independent,gradient_updates=updates_count,
        source_and_parent_unchanged=True,input_file_hashes_checked=len(p['input_sha256']),audit_code_sha256=sha(Path(__file__)))
    _durable_json(root/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
