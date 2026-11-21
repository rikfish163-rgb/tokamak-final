"""Independent shot counts plus same-implementation full CUDA replay/refit."""
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
from ..credit_horizon_20261023.run import READ,REFERENCE,REF_SHA,STEPS,load_model,schedule
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..horizon_memory_20261023.audit import independently_score,aggregate
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import enable
from .run import ARMS,CURVES,PLAN,gate,train


def deterministic_history(history):
    result=copy.deepcopy(history)
    for row in result:
        for a in ARMS:del row[a]['seconds']
    return result


def check_training(entries,history,fit):
    assert len(entries)==len(history)
    totals=dict(episodes=0,forward_steps=0)
    for i,(entry,row) in enumerate(zip(entries,history)):
        assert entry['step']==row['step']==i and len(entry['episodes'])==2
        assert [e['disruptive'] for e in entry['episodes']]==[False,True]
        for a in ARMS:
            assert len(row[a]['episodes'])==2 and math.isfinite(row[a]['grad_norm'])
            assert all(math.isfinite(v) and v>0 for v in row[a]['group_grad_norms'].values())
            for e,r in zip(entry['episodes'],row[a]['episodes']):
                assert e['shot'] in fit and {k:r[k] for k in e}==e
                counts=list(range(256,e['count']+1,128))
                assert r['all_counts']==counts and counts and e['count']%128==0
                assert min(e['ends'])>=e['count'] and max(e['ends'])<e['count']+128
                expected=(r['risk_before'] if a=='joint' else .5*(r['risk_before']+r['risk_after']))+.2*r['forecast']
                assert math.isclose(expected,r['objective'],rel_tol=2e-6,abs_tol=2e-6)
                totals['episodes']+=1;totals['forward_steps']+=len(counts)
    return totals


def audit(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    assert done['status']=='COMPLETED' and not done['test360_accessed']
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    assert sha(PLAN)==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256==p['source_sha256']
    assert sha(REFERENCE/'completed.json')==REF_SHA==p['reference_sha256']
    assert p['folds']==READ(REFERENCE/'protocol.json')['folds']
    torch.set_num_threads(2);torch.manual_seed(131)
    scores={a:[] for a in CURVES};details=[];access=[];replay=[];training={};scope={}
    stats={a:dict(steps=0,loss_decreased=0,max_step=0.) for a in ARMS}
    seals={(r['fold'],r['shot']):r for r in READ(root/'seals.json')}
    original_signal,original_outcome=simple_data.read_shot,io23.read_outcome
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';phase='fit'
        assert len(plan['fit'])==15 and len(plan['held'])==5 and not set(plan['fit'])&set(plan['held'])
        def signal_guard(device,sid,**kwargs):
            assert device=='EAST' and sid in plan['fit']+plan['held']
            access.append(dict(fold=f,kind='signal',shot=sid,phase=phase));return original_signal(device,sid,**kwargs)
        def outcome_guard(device,sid):
            assert device=='EAST' and sid in plan['fit'] and phase=='fit'
            access.append(dict(fold=f,kind='outcome',shot=sid,phase=phase));return original_outcome(device,sid)
        warm=load_model(f);original=state_cpu(warm);allowed=enable(warm);warm.set_stage('frozen')
        mapping=READ(folder/'input_map.json');assert mapping==READ(REFERENCE/f'fold{f}'/'input_map.json')
        models={};frozen={};scope[f'fold{f}']={}
        for a in ARMS:
            m=copy.deepcopy(warm);m.load_state_dict(torch.load(folder/f'{a}.pt',map_location='cuda',weights_only=True)['model'])
            m.set_stage('frozen');assert_unchanged(m,original,tuple(allowed));models[a]=m;frozen[a]=state_cpu(m)
            changes={k:float((v.double()-original[k].double()).norm()) for k,v in frozen[a].items() if k in allowed}
            assert any(changes[k]>0 for k in m.encoder_names())
            assert any(changes[k]>0 for k in changes if k.startswith('base.forecast_head.'))
            scope[f'fold{f}'][a]=changes
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
            shots=[]
            for sid in plan['fit']:
                s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
                shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
            entries,eligible=schedule(shots,plan['fit'],f)
            assert len(entries)==STEPS and entries==READ(folder/'schedule.json') and eligible==READ(folder/'eligibility.json')
            training[f'fold{f}']=check_training(entries,READ(folder/'training.json'),set(plan['fit']))
            if f==0:
                out=root/'audit_refit0';out.mkdir();refit,history=train(warm,shots,entries,p['threshold'],out,f)
                assert deterministic_history(history)==deterministic_history(READ(folder/'training.json'))
                for a in ARMS:assert_unchanged(refit[a],frozen[a])
                del refit
            phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                path=folder/'shots'/f'{sid}.npz';assert sha(path)==seals[f,sid]['sha256']
                updates=READ(folder/'shots'/f'{sid}_updates.json')
                with np.load(path) as stored,np.load(REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz') as old:
                    np.testing.assert_array_equal(stored['times'],s['times'])
                    np.testing.assert_array_equal(stored['warm_static'],old['static'])
                    np.testing.assert_array_equal(stored['warm_intra'],old['increment'])
                    for a,m in models.items():
                        curves,receipts,packets=predict(m,signal)
                        assert receipts==updates[a]
                        for k,v in curves.items():np.testing.assert_array_equal(v,stored[f'{a}_{k}'])
                        np.testing.assert_array_equal(stored[f'{a}_intra'][:256],stored[f'{a}_static'][:256])
                        with np.load(folder/'shots'/f'{sid}_{a}_forecasts.npz') as forecast:
                            for k,v in packets.items():np.testing.assert_array_equal(v,forecast[k])
                        assert len(receipts)==len(range(256,len(s['times']),128))
                        for j,r in enumerate(receipts):
                            count=256+j*128
                            assert r['last_observation']==count-1 and r['first_affected_prediction']==count
                            assert all(e+25<=count-1 for e in r['forecast_origins'])
                            assert r['parameter_delta_norm']<=.00101
                            stats[a]['steps']+=r['gradient_steps']
                            stats[a]['loss_decreased']+=int(r.get('loss_after',float('inf'))<r.get('loss_before',-float('inf')))
                            stats[a]['max_step']=max(stats[a]['max_step'],r['parameter_delta_norm'])
                replay.append(dict(fold=f,shot=sid,exact=True));print(dict(stage='replay',shot=sid,done=len(replay)),flush=True)
            for a,m in models.items():assert_unchanged(m,frozen[a])
            assert_unchanged(warm,original)
        # Evaluation cannot affect replayed parameters or sealed predictions.
        for sid in plan['held']:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            v=s['valid'];timely=v&(np.round((s['down']-s['times'])*1000,9)>30.) if s['is_disrupt'] else v
            with np.load(folder/'shots'/f'{sid}.npz') as z:
                for a in CURVES:
                    r=independently_score(sid,s['is_disrupt'],s['down'],s['times'][v],z[a][v],p['threshold'])
                    scores[a].append(r);details.append(dict(fold=f,arm=a,**r,
                        max_probability=float(z[a][v].max()),max_timely_probability=float(z[a][timely].max()) if timely.any() else None))
        del models,warm
    assert access==READ(root/'access.json') and len(replay)==len({r['shot'] for r in replay})==20
    assert READ(root/'evaluation_access.json')==[dict(shot=sid,phase='after_all_seals') for f in p['folds'] for sid in f['held']]
    result={a:aggregate(v,p['threshold']) for a,v in scores.items()};assert result==done['metrics']
    for a in CURVES:assert scores[a]==READ(root/'per_shot.json')[a]['predictions']
    assert gate(result)==done['passes_screen']
    assert sum(r['forward_steps'] for r in training.values())==done['counts']['offline_forward']
    assert sum(r['steps'] for r in stats.values())==done['counts']['online']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'diagnostic_details.json',dict(rows=details,updates=stats,parameter_changes=scope))
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        diagnostic_sha256=sha(root/'diagnostic_details.json'),metrics=result,gate=gate(result),
        independent_shot_counts=120,full_replayed_shots=20,new_models_per_shot=2,refitted_fold=0,
        exact_training_values_excluding_timing=True,training=training,updates=stats,
        source_and_frozen_readouts_unchanged=True,test360_accessed=False,seconds=time.monotonic()-start,
        limitation='Four disruptive development identities; neural replay uses same implementation; not independent future confirmation.')
    _durable_json(root/'verification.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
