"""Recount predictions and verify matched draws, causal replay and a full refit."""
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
from ..horizon_memory_20261023.audit import independently_score,aggregate,check_manifests
from ..horizon_memory_20261023.learning import FoldAccess
from ..paired_feedback_20261022.model import PairedFeedback
from ..preparation_factorial_20261023.learning import mapping_for,predict
from ..redesign_20261004.prepare_prior import load_parent,load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import fit_statistics,transform,SIGNAL_KEYS
from .learning import POLICIES
from .prepare import prepare

READ=lambda p:json.loads(p.read_text())
LINES=lambda p:[json.loads(line) for line in p.read_text().splitlines()]


def contrasts(stats):
    names={f'{c}_{a}' for c in POLICIES for a in ('static','intra')}
    if set(stats)!=names or any(r['n_disrupt']!=4 or r['n_safe']!=16 for r in stats.values()):
        raise ValueError('four arms on four disruptive and sixteen safe held shots required')
    result={}
    reference=stats['legacy_static']
    for c in POLICIES:
        s,i=stats[f'{c}_static'],stats[f'{c}_intra']
        own=i['tp']>=s['tp'] and i['fp']<=s['fp'] and i['J']>s['J'] and i['fpr']<=.2
        reference_pass=i['tp']>=reference['tp'] and i['fp']<=reference['fp'] and i['J']>reference['J']
        result[c]=dict(delta_tp=i['tp']-s['tp'],delta_fp=i['fp']-s['fp'],delta_J=i['J']-s['J'],
            own_parent_pass=bool(own),screen_pass=bool(own and reference_pass),
            delta_J_vs_legacy_static=i['J']-reference['J'])
    return result


def check_replacements(folder,records):
    """Check every original/actual pair, not merely aggregate class counts."""
    by_id={s['shot']:s for s in records}
    old=LINES(folder.parent/'legacy'/'risk_training_manifest.jsonl')
    actual=LINES(folder/'risk_training_manifest.jsonl')
    ledger=LINES(folder/'replacement_manifest.jsonl');assert len(old)==len(actual)==len(ledger)==512
    changed=0
    for step,(o,a,l) in enumerate(zip(old,actual,ledger)):
        assert o['step']==a['step']==l['step']==step
        assert a['source']==o['source'] and l['proposed']==o['target']
        expected=[]
        for j,(before,after) in enumerate(zip(o['target'],a['target'])):
            replace=folder.name=='safe_negative' and before['label']==0 and by_id[before['shot']]['is_disrupt']
            if replace:
                s=by_id[after['shot']]
                assert not s['is_disrupt'] and after['label']==0 and s['valid'][after['end']]
                assert s['y'][after['end']]==0
                expected.append(dict(index=j,before=before,after=after))
            else:assert before==after
        assert l['replaced']==expected;changed+=len(expected)
    assert changed==READ(folder/'preparation_receipt.json')['replaced_negative_draws']
    return changed


def audit(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    p,done=READ(root/'protocol.json'),READ(root/'completed.json')
    assert done['status']=='COMPLETED' and not done['test360_accessed']
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    assert sha(Path(p['plan_path']))==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256==p['source_sha256']
    torch.set_num_threads(2);torch.manual_seed(131)
    base,info=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda();parent.set_stage('frozen')
    origin=state_cpu(parent);assert float(info['threshold'])==p['threshold']
    source_ids=io23.source_roles()['train'];read_signal,read_outcome=simple_data.read_shot,io23.read_outcome
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    source_stats=fit_statistics([{k:s[k] for k in SIGNAL_KEYS} for s in source],device='J-TEXT',allowed_ids=source_ids)
    assert source_stats==READ(root/'source_statistics.json') and source_ids==READ(root/'source_label_access.json')
    summaries={f'{c}_{a}':[] for c in POLICIES for a in ('static','intra')}
    seals={(r['fold'],r['policy'],r['shot']):r for r in READ(root/'prediction_receipts.json')}
    replay=[];counts={};replacement_counts={};steps=0;details=[]
    for plan in p['folds']:
        f=plan['fold']
        for c in POLICIES:
            folder=root/f'fold{f}'/c;access=FoldAccess(plan,source_ids)
            def signal_guard(device,sid,**kwargs):
                access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
            def outcome_guard(device,sid):
                access.check('outcome',device,sid);return read_outcome(device,sid)
            saved=torch.load(folder/'parent.pt',map_location='cuda',weights_only=True)
            model=copy.deepcopy(parent);model.load_state_dict(saved['model']);model.set_stage('frozen')
            assert_unchanged(model,origin,tuple(model.encoder_names())+('base.forecast_head.weight','base.forecast_head.bias'))
            frozen=state_cpu(model)
            with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
                records=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in plan['fit']]
                mapping=mapping_for(source_stats,records,plan['fit'],'raw_bce');assert mapping==READ(folder/'input_map.json')
                target=[dict(r,**transform({k:r[k] for k in ('values','available')},mapping)) for r in records]
                counts[f'{f}/{c}']=check_manifests(folder,source,target,set(plan['fit']))
                replacement_counts[f'{f}/{c}']=check_replacements(folder,target)
                assert (folder/'forecast_manifest.jsonl').read_bytes()==(folder.parent/'legacy'/'forecast_manifest.jsonl').read_bytes()
                access.phase='predict'
                for sid in plan['held']:
                    s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                    scores,updates=predict(model,transform({k:s[k] for k in ('values','available')},mapping))
                    path=folder/'shots'/f'{sid}.npz';up=folder/'shots'/f'{sid}_updates.json';seal=seals[f,c,sid]
                    assert sha(path)==seal['prediction_sha256'] and sha(up)==seal['updates_sha256'] and updates==READ(up)
                    with np.load(path,allow_pickle=False) as trace:
                        np.testing.assert_array_equal(trace['times'],s['times'])
                        for a in ('static','intra'):np.testing.assert_array_equal(scores[a],trace[a])
                        np.testing.assert_array_equal(trace['static'][:256],trace['intra'][:256])
                    for j,r in enumerate(updates):
                        assert r['last_observation']==255+128*j and r['first_affected_prediction']==256+128*j
                        assert r['first_affected_prediction']<len(s['times']) and r['parameter_delta_norm']<=.00101
                        assert all(e+25<=r['last_observation'] for e in r['forecast_origins'])
                        steps+=r['gradient_steps']
                    replay.append(dict(fold=f,policy=c,shot=sid,exact=True))
                assert_unchanged(model,frozen)
            assert access.log==READ(folder/'access.json')
            for sid in plan['held']:
                s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
                with np.load(folder/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                    for a in ('static','intra'):
                        valid=s['valid'];r=independently_score(sid,s['is_disrupt'],s['down'],s['times'][valid],trace[a][valid],p['threshold'])
                        summaries[f'{c}_{a}'].append(r);details.append(dict(fold=f,policy=c,mode=a,**r))
            print(dict(stage='verified',fold=f,policy=c,streams=len(replay)),flush=True)
    results={a:aggregate(rows,p['threshold']) for a,rows in summaries.items()}
    assert results==done['metrics'] and steps==done['neural_steps']
    expected=READ(root/'per_shot.json')
    for a in summaries:assert summaries[a]==expected[a]['predictions']
    # One deterministic candidate refit checks reproducibility, not significance.
    plan=p['folds'][0];out=root/'audit_refit0_safe_negative';out.mkdir();access=FoldAccess(plan,source_ids)
    def signal_guard(device,sid,**kwargs):
        access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
    def outcome_guard(device,sid):
        access.check('outcome',device,sid);return read_outcome(device,sid)
    with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
        refit,_,_=prepare(parent,plan['fit'],source,source_stats,out,'safe_negative')
    ref=root/'fold0'/'safe_negative';weights=torch.load(ref/'parent.pt',map_location='cpu',weights_only=True)['model']
    for k,v in refit.state_dict().items():torch.testing.assert_close(v.cpu(),weights[k],rtol=0,atol=0)
    for name in ('input_map.json','risk_training_manifest.jsonl','replacement_manifest.jsonl','forecast_manifest.jsonl','risk_training.json','forecast_training.json'):
        assert (out/name).read_bytes()==(ref/name).read_bytes(),name
    assert_unchanged(parent,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    analysis=dict(contrasts=contrasts(results),per_shot=details)
    _durable_json(root/'analysis.json',analysis)
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        test_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_east_negative_sampling_audit.py'),
        analysis_sha256=sha(root/'analysis.json'),recounts=len(details),full_stream_replays=replay,
        exact_refits=[dict(fold=0,policy='safe_negative')],training_draws_checked=counts,replacements=replacement_counts,
        source_and_frozen_parameters_verified=True,actual_intra_steps=steps,metrics=results,
        scope='same implementation replay; development CV20; no independent future confirmation')
    _durable_json(root/'verification.json',report)
    print(dict(status='VERIFIED',recounts=len(details),replays=len(replay),contrasts=analysis['contrasts']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
