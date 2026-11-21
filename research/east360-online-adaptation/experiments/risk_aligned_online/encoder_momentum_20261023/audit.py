"""Recount all shots, replay both models, and refit fold0 with source replay."""
import argparse
import copy
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ,REFERENCE,REF_SHA,load_model,schedule
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..east_joint_feedback_20261023.audit import check_training,deterministic_history
from ..east_joint_feedback_20261023.learning import enable
from ..horizon_memory_20261023.audit import independently_score,aggregate
from ..increment_forecast_20261023.learning import predict
from ..labeled_alignment_20261023.learning import balanced_batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import pools,sha
from ..warmstart_alignment_20261022.alignment import transform
from ..east_joint_source_replay_20261023.learning import replay_loss,validate_source
from .run import ARMS,CURVES,PREVIOUS,PREVIOUS_SHA,PLAN,gate,train


def audit(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    assert sha(PLAN)==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256==p['source_sha256']
    assert sha(PREVIOUS/'completed.json')==PREVIOUS_SHA==p['previous_sha256']
    assert sha(REFERENCE/'completed.json')==REF_SHA==p['reference_sha256']
    assert p['folds']==READ(PREVIOUS/'protocol.json')['folds']
    torch.set_num_threads(2);torch.manual_seed(131)
    source_ids=io23.source_roles()['train'];source_access=[];original_outcome=io23.read_outcome
    def guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        source_access.append(sid);return original_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guard):source,ids=load_source_train()
    validate_source(source,source_ids)
    assert ids==source_access==p['source_train_ids']==READ(root/'source_label_access.json')
    source_byid={s['shot']:s for s in source};groups=pools(source)
    scores={a:[] for a in CURVES};rows=[];training={};retention={};updates={a:0 for a in ARMS};access=[]
    original_signal=simple_data.read_shot;seals={(r['fold'],r['shot']):r for r in READ(root/'seals.json')}
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';phase='fit'
        assert len(plan['fit'])==15 and len(plan['held'])==5 and not set(plan['fit'])&set(plan['held'])
        def signal_guard(device,sid,**kwargs):
            assert device=='EAST' and sid in plan['fit']+plan['held']
            access.append(dict(fold=f,kind='signal',shot=sid,phase=phase));return original_signal(device,sid,**kwargs)
        def outcome_guard(device,sid):
            assert device=='EAST' and sid in plan['fit'] and phase=='fit'
            access.append(dict(fold=f,kind='outcome',shot=sid,phase=phase));return original_outcome(device,sid)
        warm=load_model(f);before=state_cpu(warm);allowed=enable(warm);warm.set_stage('frozen')
        models={};frozen={};mapping=READ(folder/'input_map.json')
        assert mapping==READ(REFERENCE/f'fold{f}'/'input_map.json')
        for a in ARMS:
            m=copy.deepcopy(warm);m.load_state_dict(torch.load(folder/f'{a}.pt',map_location='cuda',weights_only=True)['model'])
            m.set_stage('frozen');assert_unchanged(m,before,tuple(allowed));models[a]=m;frozen[a]=state_cpu(m)
            assert any(not torch.equal(frozen[a][k],before[k]) for k in m.encoder_names())
            assert not torch.equal(frozen[a]['base.forecast_head.weight'],before['base.forecast_head.weight'])
        history=READ(folder/'training.json');rng=np.random.default_rng(9131+f)
        replay_bce={a:[] for a in ('warm',)+ARMS};checkpoint_retention=READ(folder/'source_training_retention.json')
        for r in history:
            sx,sm,sy,sr=balanced_batch(source,groups,rng,warm.cfg)
            assert sr==r['source_examples'] and len(sr)==32
            for e in sr:
                s=source_byid[e['shot']];assert s['valid'][e['end']] and int(s['y'][e['end']]>.5)==e['label']
            with torch.no_grad():
                for a,m in dict(warm=warm,**models).items():replay_bce[a].append(float(replay_loss(m,sx,sm,sy)))
            assert replay_bce['warm'][-1]==r['warm_source_bce']
            for a in ARMS:
                expected=r[a]['source_bce']+sum(v['objective'] for v in r[a]['episodes'])/2
                assert math.isclose(expected,r[a]['objective'],rel_tol=1e-7)
        assert replay_bce==checkpoint_retention
        retention[f'fold{f}']={a:float(np.mean(v)) for a,v in replay_bce.items()}
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
            shots=[]
            for sid in plan['fit']:
                s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
                shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
            entries,eligible=schedule(shots,plan['fit'],f)
            assert entries==READ(folder/'schedule.json')==READ(PREVIOUS/f'fold{f}'/'schedule.json')
            assert eligible==READ(folder/'eligibility.json')
            training[f'fold{f}']=check_training(entries,history,set(plan['fit']))
            if f==0:
                out=root/'audit_refit0';out.mkdir();refit,refit_history=train(warm,shots,entries,p['threshold'],out,f,source)
                assert deterministic_history(refit_history)==deterministic_history(history)
                for a in ARMS:assert_unchanged(refit[a],frozen[a])
                del refit
            phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                path=folder/'shots'/f'{sid}.npz';assert sha(path)==seals[f,sid]['sha256']
                receipts=READ(folder/'shots'/f'{sid}_updates.json')
                with np.load(path) as z,np.load(REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz') as old:
                    np.testing.assert_array_equal(z['times'],s['times'])
                    np.testing.assert_array_equal(z['warm_static'],old['static'])
                    np.testing.assert_array_equal(z['warm_intra'],old['increment'])
                    for a,m in models.items():
                        pred,records,packets=predict(m,signal);assert records==receipts[a]
                        for k,v in pred.items():np.testing.assert_array_equal(v,z[f'{a}_{k}'])
                        np.testing.assert_array_equal(z[f'{a}_intra'][:256],z[f'{a}_static'][:256])
                        with np.load(folder/'shots'/f'{sid}_{a}_forecasts.npz') as stored:
                            for k,v in packets.items():np.testing.assert_array_equal(v,stored[k])
                        assert len(records)==len(range(256,len(s['times']),128))
                        for j,r in enumerate(records):
                            count=256+j*128
                            assert r['last_observation']==count-1 and r['first_affected_prediction']==count
                            assert all(e+25<=count-1 for e in r['forecast_origins']) and r['parameter_delta_norm']<=.00101
                            updates[a]+=r['gradient_steps']
                print(dict(stage='replay',fold=f,shot=sid),flush=True)
            for a,m in models.items():assert_unchanged(m,frozen[a])
            assert_unchanged(warm,before)
        for sid in plan['held']:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            valid=s['valid'];timely=valid&(np.round((s['down']-s['times'])*1000,9)>30.) if s['is_disrupt'] else valid
            with np.load(folder/'shots'/f'{sid}.npz') as z:
                for a in CURVES:
                    r=independently_score(sid,s['is_disrupt'],s['down'],s['times'][valid],z[a][valid],p['threshold'])
                    scores[a].append(r);rows.append(dict(fold=f,arm=a,**r,max_probability=float(z[a][valid].max()),
                        max_timely_probability=float(z[a][timely].max()) if timely.any() else None))
        del models,warm
    result={a:aggregate(v,p['threshold']) for a,v in scores.items()};assert result==done['metrics']
    for a in CURVES:assert scores[a]==READ(root/'per_shot.json')[a]['predictions']
    assert access==READ(root/'access.json') and gate(result)==done['passes_screen']
    assert sum(r['forward_steps'] for r in training.values())==done['counts']['offline_forward']
    assert sum(updates.values())==done['counts']['online'] and done['counts']['source_windows']==64*32*4*2
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'diagnostic_details.json',dict(rows=rows,source_training_bce=retention))
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        metrics=result,passes_screen=gate(result),independent_shot_counts=120,full_replayed_shots=20,
        refitted_fold=0,source_training_bce=retention,source_and_frozen_readouts_unchanged=True,
        test360_accessed=False,seconds=time.monotonic()-start,
        limitation='Four disruptive development identities; same-implementation neural replay; source BCE is training retention not test performance.')
    _durable_json(root/'verification.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)

