"""Verify held identity boundaries, real gradients and exact deployed replays."""
import argparse
import gc
import json
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..finite_conditional_loss_20261025.learning import load_policy,policy_step,outer_gradient
from ..history_tangent_20261023.learning import logit,flat,apply_step
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import SIGNAL_KEYS
from ..warmstart_alignment_20261022.run import predict_many
from ..operational_bag_loss_20261025.learning import bag_loss
from .design import fit_fold_map
from .learning import ARMS,evaluate_teachers,predict,eligible
from .fit import TEACHERS,PREVIOUS,load_teachers,cache_teachers


def audit_teachers():
    done=READ(TEACHERS/'completed.json');p=READ(TEACHERS/'protocol.json')
    for name,digest in done['artifact_sha256'].items():assert sha(TEACHERS/name)==digest,name
    for path,digest in p['input_sha256'].items():assert sha(Path(path))==digest,path
    source_ids=set(io23.source_roles()['train']);records={};seen=set();counts=[]
    for sid in p['preparation_ids']:
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        records[sid]={k:raw[k] for k in SIGNAL_KEYS}
    for fold in p['folds']:
        assert not seen.intersection(fold['held']);seen.update(fold['held'])
        assert set(fold['train']).isdisjoint(fold['held'])
        assert set(fold['train']+fold['held'])==set(p['preparation_ids'])
        folder=TEACHERS/f'fold{fold["fold"]}';mapping=READ(folder/'input_map.json')
        assert set(mapping['source_statistics']['shot_ids'])==source_ids
        assert fit_fold_map(mapping['source_statistics'],list(records.values()),fold['train'])==mapping
        step_counts={}
        for name,expected in (('risk_manifest.jsonl',512),('forecast_manifest.jsonl',256)):
            rows=[json.loads(line) for line in (folder/name).read_text().splitlines()]
            assert len(rows)==expected
            for row in rows:
                assert {r['shot'] for r in row['target']}<=set(fold['train'])
                assert {r['shot'] for r in row['source']}<=source_ids
            step_counts[name]=len(rows)
        counts.append(dict(fold=fold['fold'],steps=step_counts,density_gain=mapping['gain'][12]))
    assert seen==set(p['preparation_ids'])
    assert not seen.intersection(p['excluded_calibration_ids']+p['excluded360_ids'])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==p['deployed_parent_sha256']
    result=dict(status='passed',completed_sha256=sha(TEACHERS/'completed.json'),
        folded_training_steps=3072,folds=counts,all20_held_once=True,
        input_hashes_checked=20,training_target_ids_strictly_exclude_held=True,
        density_maps_recomputed_from_training15_only=True,source_and_deployed_parent_unchanged=True,
        audit_code_sha256=sha(Path(__file__)))
    _durable_json(TEACHERS/'audit.json',result)
    return result


def main(root):
    teacher_audit=audit_teachers()
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
                assert len(ds)==(4 if flag else 16)
                total+=sum(d+max(d,0) for d in ds)/(2*len(ds))
            np.testing.assert_allclose(total,r['objective'],atol=1e-12,rtol=1e-10)
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    teachers=load_teachers(parent);mapping=READ(REFERENCE/'input_map.json')
    def deny(*args,**kwargs):raise AssertionError('new outcomes forbidden during audit')
    with patch.object(io23,'read_outcome',side_effect=deny):
        teacher_replays=[]
        for teacher in teachers:
            sid=next(s for s in teacher['held'] if s in (63226,70868,75604,88785))
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},teacher['mapping'])
            scores=predict_many(teacher['model'],{'static':s},device='cuda')['static']
            with torch.no_grad():fused=predict_many(teacher['model'],{'static':s},device='cuda')['static']
            with np.load(TEACHERS/f'fold{teacher["fold"]}'/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(scores,z['math_scores']);np.testing.assert_array_equal(fused,z['scores'])
            teacher_replays.append(sid)
        receipts,baselines,_=cache_teachers(teachers,p['threshold'])
        assert receipts==READ(root/'cache_receipts.json')
        assert baselines==READ(root/'historical_baselines.json')
        objective_replays={}
        for kind in ('global','context'):
            policy=load_policy(records[kind],device='cuda')
            loss,g,detail,gate=evaluate_teachers(teachers,policy,p['threshold'])
            r=best[kind]
            assert loss==r['objective'] and detail==r['per_shot'] and gate==r['history_gate']
            assert float(np.linalg.norm(g))==r['gradient_norm']
            objective_replays[kind]=dict(epoch=r['epoch'],objective=loss,history_gate=gate)
            print(dict(stage='selected_objective_replayed',kind=kind,objective=loss),flush=True)
        # A max can move into the unchanged first256 points. Such a shot has
        # exactly zero policy derivative, not a broken gradient calculation.
        # Probe the first disruptive winner with an actual update, and record
        # all zero-credit prefixes explicitly instead of silently skipping them.
        zero_credit=[r['shot'] for r in best['context']['per_shot'] if r['count']<256]
        candidates=sorted((r for r in best['context']['per_shot'] if r['flag'] and r['count']>=256),key=lambda r:r['shot'])
        if not candidates:raise AssertionError('no disruptive updated winner for real-model chain verification')
        row=candidates[0];sid=row['shot']
        teacher=next(t for t in teachers if sid in t['held']);shot=teacher['shots'][sid]
        item=shot['cache'][row['count']];fast=initial_encoder(teacher['model'])
        parts=item['parts'].to('cuda');c=item['context'].to('cuda');observed=item['observed'].to('cuda')
        policy=load_policy(records['context'],device='cuda');step=policy_step(parts,policy(c,observed))
        ref={};offset=0
        for k,v in fast.items():
            n=v.numel();ref[k]=v.detach()-step[offset:offset+n].reshape_as(v);offset+=n
        windows=[simple_data.window(shot['signal'],int(e),128) for e in row['chunk']]
        x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
        z=logit(teacher['model'],x,m,ref)[row['position']]
        reference=flat(torch.autograd.grad(bag_loss(z,True,p['threshold']),tuple(policy.parameters()))).detach()
        changed=apply_step(fast,step.detach());z=logit(teacher['model'],x,m,changed)[row['position']]
        risk=flat(torch.autograd.grad(bag_loss(z,True,p['threshold']),tuple(changed.values()))).detach()
        analytic=outer_gradient(policy,c,observed,parts,item['scale'],risk)
        torch.testing.assert_close(analytic,reference,atol=1e-9,rtol=2e-5)
        chain=dict(shot=sid,count=row['count'],end=row['end'],
                   selection='smallest disruptive ID whose winning endpoint depends on update',
                   static_prefix_zero_policy_derivative=zero_credit,
                   max_absolute_error=float((analytic-reference).abs().max()),reference_norm=float(reference.norm()))
        del teachers,teacher,shot;gc.collect()
        replays=[];insample=READ(PREVIOUS/'policies.json')['context']
        for sid in (89272,89284,89584):
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,s,records,insample)
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
                (danger if o['is_disrupt'] else safe).append(float(np.logaddexp(0,b-zpeak if o['is_disrupt'] else zpeak-b)))
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
    report=dict(status='passed',completed_sha256=sha(root/'completed.json'),teacher_audit=teacher_audit,
        teachers_replayed_both_backends=teacher_replays,training_objective_recounts=22,
        cached_blocks_rebuilt=len(receipts),selected_objectives_replayed=objective_replays,real_model_chain_rule=chain,
        complete_shots_replayed=replays,independent_metrics=independent,gradient_updates=updates_count,
        source_and_parent_unchanged=True,input_file_hashes_checked=len(p['input_sha256']),audit_code_sha256=sha(Path(__file__)))
    _durable_json(root/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
