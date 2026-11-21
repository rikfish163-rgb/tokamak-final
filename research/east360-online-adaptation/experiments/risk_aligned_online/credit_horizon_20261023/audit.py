"""Recount warnings independently, replay all held streams and refit one fold.

Neural replay uses the deployed implementation: it tests reproducibility, not
an independent implementation of the model. Warning counts are independent.
"""
import argparse
import copy
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..horizon_memory_20261023.audit import independently_score, aggregate
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import ARMS, PLAN, READ, REFERENCE, REF_SHA, STEPS, load_model, schedule, train


def check_training(entries, history, fit):
    assert len(entries) == len(history)
    totals = dict(episodes=0, forward_steps=0, differentiated_steps=0)
    for step, (entry, row) in enumerate(zip(entries, history)):
        assert entry['step'] == row['step'] == step
        assert len(entry['episodes']) == 2
        assert [e['disruptive'] for e in entry['episodes']] == [False, True]
        for arm in ARMS:
            receipts = row[arm]['episodes']
            assert len(receipts) == 2
            assert math.isfinite(row[arm]['outer_gradient_norm']) and row[arm]['outer_gradient_norm'] >= 0
            for e, r in zip(entry['episodes'], receipts):
                assert e['shot'] in fit and {k:r[k] for k in e} == e
                counts = list(range(256, e['count']+1, 128))
                assert counts and e['count'] % 128 == 0
                assert e['ends'] == sorted(set(e['ends']))
                assert e['count'] <= min(e['ends']) <= max(e['ends']) < e['count']+128
                assert r['all_counts'] == counts and r['differentiated'] == counts[-int(arm[-1]):]
                assert all(math.isfinite(r[k]) for k in ('risk_before','risk_after','forecast','objective'))
                expected = r['forecast']+(r['risk_after']-r['risk_before'])/.001
                # Receipts serialize float32 arithmetic and operands separately.
                assert math.isclose(r['objective'], expected, rel_tol=2e-5, abs_tol=2e-5)
                totals['episodes'] += 1
                totals['forward_steps'] += len(counts)
                totals['differentiated_steps'] += len(r['differentiated'])
    return totals


def audit(root):
    if (root/'verification.json').exists():
        raise FileExistsError(root/'verification.json')
    start = time.monotonic()
    p, done = READ(root/'protocol.json'), READ(root/'completed.json')
    assert done['status'] == 'COMPLETED' and not done['test360_accessed']
    assert sha(REFERENCE/'completed.json') == REF_SHA == p['reference_sha256']
    assert sha(PLAN) == p['plan_sha256'] and sha(DEFAULT_PARENT) == PARENT_SHA256 == p['source_sha256']
    for name, digest in done['artifact_sha256'].items(): assert sha(root/name) == digest, name
    for name, digest in p['code_sha256'].items(): assert sha(io23.ROOT/name) == digest, name
    assert p['folds'] == READ(REFERENCE/'protocol.json')['folds']
    torch.set_num_threads(2); torch.manual_seed(131)
    arms = ('static','warm_intra')+ARMS
    rows = {a:[] for a in arms}; details=[]; expected_access=[]; training_counts={}; replay=[]
    stats = {a:dict(steps=0,loss_decreased=0,max_step=0.) for a in ARMS}
    seals = {(r['fold'],r['shot']):r for r in READ(root/'seals.json')}
    original_signal, original_outcome = simple_data.read_shot, io23.read_outcome
    for plan in p['folds']:
        f=plan['fold']; folder=root/f'fold{f}'; phase='fit'
        assert len(plan['fit'])==15 and len(plan['held'])==5 and not set(plan['fit']) & set(plan['held'])
        def signal_guard(device, sid, **kwargs):
            assert device == 'EAST' and sid in plan['fit']+plan['held']
            expected_access.append(dict(fold=f,kind='signal',shot=sid,phase=phase))
            return original_signal(device,sid,**kwargs)
        def outcome_guard(device, sid):
            assert device == 'EAST' and sid in plan['fit'] and phase == 'fit'
            expected_access.append(dict(fold=f,kind='outcome',shot=sid,phase=phase))
            return original_outcome(device,sid)
        warm = load_model(f); original = state_cpu(warm)
        mapping = READ(folder/'input_map.json')
        assert mapping == READ(REFERENCE/f'fold{f}'/'input_map.json')
        models={}
        for a in ARMS:
            models[a]=copy.deepcopy(warm)
            models[a].load_state_dict(torch.load(folder/f'{a}.pt',map_location='cuda',weights_only=True)['model'])
            models[a].set_stage('frozen')
            assert_unchanged(models[a],original,('base.forecast_head.weight','base.forecast_head.bias'))
            assert any(not torch.equal(v.cpu(),original[k]) for k,v in models[a].state_dict().items())
        frozen={a:state_cpu(m) for a,m in models.items()}
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
            shots=[]
            for sid in plan['fit']:
                s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
                shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
            entries, eligible = schedule(shots,plan['fit'],f)
            assert len(entries)==STEPS and entries==READ(folder/'schedule.json')
            assert eligible==READ(folder/'eligibility.json')
            training_counts[f'fold{f}']=check_training(entries,READ(folder/'training.json'),set(plan['fit']))
            if f==0:
                refit_folder=root/'audit_refit0';refit_folder.mkdir()
                refit, history=train(warm,shots,entries,p['threshold'],refit_folder,f)
                assert history==READ(folder/'training.json')
                for a in ARMS:
                    assert_unchanged(refit[a],frozen[a])
                del refit
            phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                path=folder/'shots'/f'{sid}.npz'; assert sha(path)==seals[(f,sid)]['sha256']
                saved_updates=READ(folder/'shots'/f'{sid}_updates.json')
                with np.load(path) as trace, np.load(REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz') as prior:
                    np.testing.assert_array_equal(trace['times'],s['times'])
                    np.testing.assert_array_equal(trace['static'],prior['static'])
                    np.testing.assert_array_equal(trace['warm_intra'],prior['increment'])
                    for a, model in models.items():
                        scores, updates, packets = predict(model,signal)
                        assert updates==saved_updates[a]
                        np.testing.assert_array_equal(scores['static'],trace['static'])
                        np.testing.assert_array_equal(scores['intra'],trace[a])
                        np.testing.assert_array_equal(trace[a][:256],trace['static'][:256])
                        with np.load(folder/'shots'/f'{sid}_{a}_forecasts.npz') as stored:
                            for k,v in packets.items():np.testing.assert_array_equal(v,stored[k])
                        assert len(updates)==len(range(256,len(s['times']),128))
                        for j,r in enumerate(updates):
                            count=256+j*128
                            assert r['last_observation']==count-1 and r['first_affected_prediction']==count
                            assert r['forecast_origins']==[count-26-25*i for i in range(4)]
                            assert max(r['forecast_origins'])+25<=r['last_observation']
                            assert 0<=r['parameter_delta_norm']<=.00101
                            stats[a]['steps']+=r['gradient_steps']
                            stats[a]['max_step']=max(stats[a]['max_step'],r['parameter_delta_norm'])
                            stats[a]['loss_decreased']+=int(r.get('loss_after',float('inf'))<r.get('loss_before',-float('inf')))
                replay.append(dict(fold=f,shot=sid,exact=True))
                print(dict(stage='replay',fold=f,shot=sid,done=len(replay)),flush=True)
            for a,m in models.items():assert_unchanged(m,frozen[a])
            assert_unchanged(warm,original)
        # These labels are read only by the evaluator after fixed replay output.
        for sid in plan['held']:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            with np.load(folder/'shots'/f'{sid}.npz') as z:
                valid=s['valid']; times=s['times'][valid]
                timely=valid & (np.round((s['down']-s['times'])*1000.,9)>30.) if s['is_disrupt'] else valid
                for a in arms:
                    r=independently_score(sid,s['is_disrupt'],s['down'],times,z[a][valid],p['threshold'])
                    rows[a].append(r)
                    details.append(dict(fold=f,arm=a,**r,max_probability=float(z[a][valid].max()),
                                        max_timely_probability=float(z[a][timely].max()) if timely.any() else None))
        del models,warm
    assert expected_access==READ(root/'access.json')
    assert READ(root/'evaluation_access.json')==[dict(shot=sid,phase='after_all_seals') for f in p['folds'] for sid in f['held']]
    assert len({r['shot'] for r in replay})==len(replay)==20
    result={a:aggregate(v,p['threshold']) for a,v in rows.items()}
    assert result==done['metrics']
    for a in arms: assert rows[a]==READ(root/'per_shot.json')[a]['predictions']
    gate=bool(result['tail8']['J']>result['static']['J'] and result['tail8']['J']>result['tail2']['J']
              and result['tail8']['tp']>=result['static']['tp'] and result['tail8']['fpr']<=.2)
    assert gate==done['passes_screen']
    assert sum(v['forward_steps'] for v in training_counts.values())+sum(v['steps'] for v in stats.values())==done['neural_steps']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'diagnostic_details.json',dict(rows=details,updates=stats,training_counts=training_counts))
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        diagnostic_sha256=sha(root/'diagnostic_details.json'),metrics=result,gate=gate,
        independent_shot_counts=80,exact_cuda_replayed_shots=20,new_arms_per_shot=2,
        refitted_folds=[0],refitted_arms=list(ARMS),exact_training_histories=True,
        training_counts=training_counts,updates=stats,source_and_classifiers_unchanged=True,
        test360_accessed=False,seconds=time.monotonic()-start,
        limitation='Four disrupted shots; developmental cross-validation, not independent future confirmation. Neural replay uses the same implementation.')
    _durable_json(root/'verification.json',report)
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
