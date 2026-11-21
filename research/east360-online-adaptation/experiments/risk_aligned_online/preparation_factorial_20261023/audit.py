"""Recount all160 predictions, replay80 streams, refit all4 conditions in fold3."""
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
from ..redesign_20261004.prepare_prior import load_parent,load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import fit_statistics,transform,SIGNAL_KEYS
from .learning import CONDITIONS,mapping_for,predict
from .prepare import prepare

READ=lambda p:json.loads(p.read_text())


def contrasts(stats):
    names={f'{c}_{a}' for c in CONDITIONS for a in ('static','intra')}
    if set(stats)!=names or any(r['n_disrupt']!=4 or r['n_safe']!=16 for r in stats.values()):
        raise ValueError('all eight conditions on four-disruptive/sixteen-safe held shots required')
    def diff(left,right):return {f'delta_{k}':stats[left][k]-stats[right][k] for k in ('tp','fp','J')}
    intra={}
    for c in CONDITIONS:
        d=diff(f'{c}_intra',f'{c}_static')
        intra[c]=dict(**d,screen_pass=bool(d['delta_tp']>=0 and d['delta_fp']<=0 and d['delta_J']>0 and stats[f'{c}_intra']['fpr']<=.2))
    density={f'{loss}_{mode}':diff(f'mapped_{loss}_{mode}',f'raw_{loss}_{mode}') for loss in ('bce','moments') for mode in ('static','intra')}
    moment={f'{mapping}_{mode}':diff(f'{mapping}_moments_{mode}',f'{mapping}_bce_{mode}') for mapping in ('raw','mapped') for mode in ('static','intra')}
    interaction={mode:{k:density[f'moments_{mode}'][k]-density[f'bce_{mode}'][k] for k in ('delta_tp','delta_fp','delta_J')} for mode in ('static','intra')}
    return dict(intra=intra,density_map=density,moment_loss=moment,interaction=interaction,
                scope='descriptive paired differences; no significance or independent confirmation claims')


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
    summaries={f'{c}_{a}':[] for c in CONDITIONS for a in ('static','intra')}
    seals={(r['fold'],r['condition'],r['shot']):r for r in READ(root/'prediction_receipts.json')}
    replay=[];counts={};step_count=0;details=[]
    for plan in p['folds']:
        f=plan['fold']
        for c in CONDITIONS:
            folder=root/f'fold{f}'/c;access=FoldAccess(plan,source_ids)
            def signal_guard(device,sid,**kwargs):
                access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
            def outcome_guard(device,sid):
                access.check('outcome',device,sid);return read_outcome(device,sid)
            saved=torch.load(folder/'parent.pt',map_location='cuda',weights_only=True)
            model=copy.deepcopy(parent);model.load_state_dict(saved['model']);model.set_stage('frozen')
            assert_unchanged(model,origin,tuple(model.encoder_names())+('base.forecast_head.weight','base.forecast_head.bias'))
            frozen=state_cpu(model)
            assert sha(folder/'parent.pt')==READ(folder/'preparation_receipt.json')['checkpoint_sha256']
            with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
                records=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in plan['fit']]
                mapping=mapping_for(source_stats,records,plan['fit'],c);assert mapping==READ(folder/'input_map.json')
                target=[dict(r,**transform({k:r[k] for k in ('values','available')},mapping)) for r in records]
                counts[f'{f}/{c}']=check_manifests(folder,source,target,set(plan['fit']))
                for name in ('risk_training_manifest.jsonl','forecast_manifest.jsonl'):
                    assert (folder/name).read_bytes()==(root/f'fold{f}'/CONDITIONS[0]/name).read_bytes()
                access.phase='predict'
                for sid in plan['held']:
                    s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                    signal=transform({k:s[k] for k in ('values','available')},mapping)
                    scores,updates=predict(model,signal)
                    path=folder/'shots'/f'{sid}.npz';up=folder/'shots'/f'{sid}_updates.json';seal=seals[f,c,sid]
                    assert sha(path)==seal['prediction_sha256'] and sha(up)==seal['updates_sha256']
                    assert updates==READ(up)
                    with np.load(path,allow_pickle=False) as trace:
                        np.testing.assert_array_equal(trace['times'],s['times'])
                        for a in ('static','intra'):np.testing.assert_array_equal(scores[a],trace[a])
                        np.testing.assert_array_equal(trace['static'][:256],trace['intra'][:256])
                    for j,r in enumerate(updates):
                        assert r['last_observation']==255+128*j and r['first_affected_prediction']==256+128*j
                        assert r['first_affected_prediction']<len(s['times']) and r['parameter_delta_norm']<=.00101
                        assert all(e+25<=r['last_observation'] for e in r['forecast_origins'])
                        step_count+=r['gradient_steps']
                    replay.append(dict(fold=f,condition=c,shot=sid,exact=True))
                assert_unchanged(model,frozen)
            assert access.log==READ(folder/'access.json')
            for sid in plan['held']:
                s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
                with np.load(folder/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                    valid=s['valid']
                    for a in ('static','intra'):
                        r=independently_score(sid,s['is_disrupt'],s['down'],s['times'][valid],trace[a][valid],p['threshold'])
                        summaries[f'{c}_{a}'].append(r);details.append(dict(fold=f,condition=c,mode=a,**r))
            print(dict(stage='verified_condition',fold=f,condition=c,replayed=len(replay)),flush=True)
    results={a:aggregate(rows,p['threshold']) for a,rows in summaries.items()}
    assert results==done['metrics'] and step_count==done['neural_steps']
    expected=READ(root/'per_shot.json')
    for a in summaries:assert summaries[a]==expected[a]['predictions']
    # All four factorial states on one fold are refit, including the missed
    # 70868 fold. This checks deterministic execution, not new statistical repeats.
    plan=p['folds'][3];refits=[]
    for c in CONDITIONS:
        out=root/f'audit_refit3_{c}';out.mkdir();access=FoldAccess(plan,source_ids)
        def signal_guard(device,sid,**kwargs):
            access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
        def outcome_guard(device,sid):
            access.check('outcome',device,sid);return read_outcome(device,sid)
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
            refit,_,_=prepare(parent,plan['fit'],source,source_stats,out,c)
        original=torch.load(root/'fold3'/c/'parent.pt',map_location='cpu',weights_only=True)['model']
        for k,v in refit.state_dict().items():torch.testing.assert_close(v.cpu(),original[k],rtol=0,atol=0)
        for name in ('input_map.json','risk_training_manifest.jsonl','forecast_manifest.jsonl','risk_training.json','forecast_training.json'):
            assert (out/name).read_bytes()==(root/'fold3'/c/name).read_bytes()
        refits.append(dict(fold=3,condition=c,exact=True))
    assert_unchanged(parent,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    analysis=contrasts(results);analysis['per_shot']=details
    _durable_json(root/'factorial_analysis.json',analysis)
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        test_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_preparation_factorial_audit.py'),
        analysis_sha256=sha(root/'factorial_analysis.json'),recounts=160,full_stream_replays=replay,exact_refits=refits,
        training_draws_checked=counts,matched_sampling_verified=True,source_and_frozen_parameters_verified=True,
        actual_intra_steps=step_count,metrics=results,scope='same implementation replay, development data; no independent future confirmation')
    _durable_json(root/'verification.json',report)
    print(dict(status='VERIFIED',recounts=160,replayed=len(replay),refits=len(refits)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
