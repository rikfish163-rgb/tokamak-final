"""Independent warning counts, complete same-code neural replay and scope checks."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..horizon_memory_20261023.audit import independently_score,aggregate
from ..horizon_memory_20261023.learning import FoldAccess
from ..increment_forecast_20261023.learning import IncrementFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS,predict

READ=lambda p:json.loads(p.read_text())


def contrasts(results):
    if set(results)!={'static','full',*ARMS} or any((r['n_disrupt'],r['n_safe'])!=(4,16) for r in results.values()):
        raise ValueError('all five matched arms, 4 disruptive and 16 safe shots required')
    q,s,c=results['qk'],results['static'],results['full_matched']
    passed=q['tp']>=s['tp'] and q['J']>s['J'] and q['fpr']<=.2
    direction=q['tp']>=c['tp'] and q['J']>c['J']
    return dict(qk_user_target=passed,qk_better_than_matched_full=direction,
                qk_attributed_screen=passed and direction,
                vs_static={a:{f'delta_{k}':r[k]-s[k] for k in ('tp','fp','J')} for a,r in results.items() if a!='static'})


def audit(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    p,done=READ(root/'protocol.json'),READ(root/'completed.json')
    assert done['status']=='COMPLETED' and not done['test360_accessed']
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256==p['source_sha256']
    reference=Path(p['reference']);assert sha(reference/'completed.json')==p['reference_sha256']
    for name,digest in READ(reference/'completed.json')['artifact_sha256'].items():assert sha(reference/name)==digest,name
    ids=[sid for f in p['folds'] for sid in f['held']]
    assert len(set(ids))==len(ids)==20
    assert all(set(f['fit'])==set(ids)-set(f['held']) for f in p['folds'])
    torch.set_num_threads(2);torch.manual_seed(131)
    base,info=load_parent(DEFAULT_PARENT);origin=state_cpu(base)
    assert info['threshold']==p['threshold'];read_signal=simple_data.read_shot
    seals={(r['fold'],r['shot']):r for r in READ(root/'prediction_receipts.json')}
    stats={a:dict(steps=0,loss_decreased=0,attention_changed=0,value_changed=0,
        max_step=0.,max_requested_step_gap=0.,total_parameter_step=0.,max_attention_change=0.,max_value_change=0.) for a in ARMS}
    replay=[]
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';ref=reference/f'fold{f}'
        receipt=READ(folder/'parent_receipt.json')
        assert Path(receipt['checkpoint'])==ref/'increment.pt' and sha(Path(receipt['checkpoint']))==receipt['sha256']
        model=IncrementFeedback(copy.deepcopy(base),'increment').cuda()
        model.load_state_dict(torch.load(ref/'increment.pt',map_location='cuda',weights_only=True)['model'])
        model.set_stage('frozen');frozen=state_cpu(model)
        mapping=READ(folder/'input_map.json');assert mapping==READ(ref/'input_map.json')
        access=FoldAccess(plan,io23.source_roles()['train']);access.phase='predict'
        def signal_guard(device,sid,**kwargs):
            access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
        def no_outcome(*args,**kwargs):raise AssertionError('labels forbidden during replay')
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=no_outcome):
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                scores,updates=predict(model,transform({k:s[k] for k in ('values','available')},mapping))
                path=folder/'shots'/f'{sid}.npz';up=folder/'shots'/f'{sid}_updates.json'
                assert sha(path)==seals[f,sid]['prediction_sha256'] and sha(up)==seals[f,sid]['updates_sha256']
                assert updates==READ(up)
                with np.load(path,allow_pickle=False) as trace:
                    np.testing.assert_array_equal(trace['times'],s['times'])
                    for a in scores:
                        np.testing.assert_array_equal(scores[a],trace[a])
                        np.testing.assert_array_equal(trace[a][:256],trace['static'][:256])
                    with np.load(ref/'shots'/f'{sid}.npz',allow_pickle=False) as prior:
                        np.testing.assert_array_equal(trace['full'],prior['increment'])
                for a,rows in updates.items():
                    for j,r in enumerate(rows):
                        assert r['last_observation']==255+128*j and r['first_affected_prediction']==256+128*j<len(s['times'])
                        assert all(e+25<=r['last_observation'] for e in r['forecast_origins'])
                        assert r['requested_step_norm']==updates['qk'][j]['requested_step_norm']
                        gap=abs(r['parameter_delta_norm']-r['requested_step_norm'])
                        assert gap<2e-7 and r['parameter_delta_norm']<=.00101
                        if a=='qk':assert r['value_change_max']==0
                        if a=='value_matched':assert r['attention_change_max']==0
                        st=stats[a];st['steps']+=r['gradient_steps']
                        st['loss_decreased']+=int(r['gradient_steps'] and r['loss_after']<r['loss_before'])
                        st['attention_changed']+=int(r['attention_change_max']>0);st['value_changed']+=int(r['value_change_max']>0)
                        st['max_step']=max(st['max_step'],r['parameter_delta_norm'])
                        st['max_requested_step_gap']=max(st['max_requested_step_gap'],gap)
                        st['total_parameter_step']+=r['parameter_delta_norm']
                        st['max_attention_change']=max(st['max_attention_change'],r['attention_change_max'])
                        st['max_value_change']=max(st['max_value_change'],r['value_change_max'])
                replay.append(dict(shot=sid,fold=f,new_arms=3,exact=True))
            assert access.log==READ(folder/'access.json');assert_unchanged(model,frozen)
        print(dict(stage='verified_fold',fold=f,shots=len(replay)),flush=True)
    rows={a:[] for a in ('static','full',*ARMS)};details=[]
    for plan in p['folds']:
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            valid=s['valid'];timely=valid&(np.round((s['down']-s['times'])*1000,9)>30.) if s['is_disrupt'] else valid
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                for a in rows:
                    r=independently_score(sid,s['is_disrupt'],s['down'],s['times'][valid],trace[a][valid],p['threshold'])
                    rows[a].append(r);details.append(dict(arm=a,**r,
                        max_timely_probability=float(trace[a][timely].max()) if timely.any() else None,
                        max_absolute_difference_from_static=float(np.abs(trace[a]-trace['static']).max())))
    results={a:aggregate(r,p['threshold']) for a,r in rows.items()}
    assert results==done['metrics'];expected=READ(root/'per_shot.json')
    for a in rows:assert rows[a]==expected[a]['predictions']
    assert sum(r['steps'] for r in stats.values())==done['neural_steps']
    assert_unchanged(base,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    analysis=dict(contrasts=contrasts(results),metrics=results,per_shot=details,updates=stats)
    _durable_json(root/'analysis.json',analysis)
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        analysis_sha256=sha(root/'analysis.json'),independent_warning_recounts=100,complete_replays=replay,
        source_unchanged=True,test360_accessed=False,
        scope='independent warning counter; same-implementation neural replay, CV20 development only')
    _durable_json(root/'verification.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
