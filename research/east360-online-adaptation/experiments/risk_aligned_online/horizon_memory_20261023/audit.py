"""Independent warning recount, all held streams replay, one complete fold refit."""
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
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import fit_statistics,make_map,transform,SIGNAL_KEYS
from .learning import ARMS,RANK_ARMS,FoldAccess,make_memory,predict,gate
from .prepare import prepare

READ=lambda p:json.loads(p.read_text())


def independently_score(sid,disrupt,down,times,prob,threshold):
    assert len(times)==len(prob) and np.isfinite(prob).all()
    first=next((i for i,v in enumerate(prob) if v>=threshold),None)
    lead=round(float(1000*(down-times[first])),9) if disrupt and first is not None else None
    return dict(shot=int(sid),is_disrupt=bool(disrupt),alarm=first is not None,
                lead_ms=lead,timely=bool(lead is not None and lead>30.))


def aggregate(rows,threshold):
    nd=sum(r['is_disrupt'] for r in rows);ns=len(rows)-nd
    tp=sum(r['timely'] for r in rows);fp=sum(r['alarm'] and not r['is_disrupt'] for r in rows)
    leads=[r['lead_ms'] for r in rows if r['timely']]
    return dict(tp=tp,fp=fp,n_disrupt=nd,n_safe=ns,tpr=tp/nd,fpr=fp/ns,J=tp/nd-fp/ns,
        threshold=threshold,median_lead_ms=float(np.median(leads)) if leads else None,
        very_early_over3000=sum(v>3000 for v in leads))


def check_manifests(folder,source,target,fit):
    groups={'source':{r['shot']:r for r in source},'target':{r['shot']:r for r in target}}
    counts={}
    for name,nsteps in (('risk_training_manifest.jsonl',512),('forecast_manifest.jsonl',256)):
        lines=[json.loads(line) for line in (folder/name).read_text().splitlines()]
        assert len(lines)==nsteps
        for step,row in enumerate(lines):
            assert row['step']==step
            for domain in groups:
                assert len(row[domain])==32
                for item in row[domain]:
                    s=groups[domain][item['shot']];e=item['end']
                    if domain=='target':assert item['shot'] in fit
                    assert 0<=e<len(s['values'])
                    if name.startswith('risk'):
                        assert s['valid'][e] and item['label']==int(s['y'][e]>.5)
                    else:assert e>=127 and e+25<len(s['values'])
        counts[name]=nsteps*64
    return counts


def audit(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    p,done=READ(root/'protocol.json'),READ(root/'completed.json')
    assert done['status']=='COMPLETED' and not done['test360_accessed']
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    assert sha(Path(p['plan_path']))==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256==p['source_sha256']
    plans=p['folds'];assert sorted(i for f in plans for i in f['held'])==p['selected_ids']
    torch.set_num_threads(2);torch.manual_seed(131)
    base,info=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda();parent.set_stage('frozen')
    original=state_cpu(parent);assert float(info['threshold'])==p['threshold']
    source_ids=io23.source_roles()['train'];source_set=set(source_ids)
    read_signal,read_outcome=simple_data.read_shot,io23.read_outcome
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_set
        return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    assert READ(root/'source_label_access.json')==source_ids
    source_stats=fit_statistics([{k:s[k] for k in SIGNAL_KEYS} for s in source],device='J-TEXT',allowed_ids=source_ids)
    assert source_stats==READ(root/'source_statistics.json')
    summaries={a:[] for a in ARMS};details=[];replay=[];training_counts={};update_stats={a:dict(steps=0,loss_decreased=0) for a in ARMS[1:]}
    selections={a:0 for a in RANK_ARMS};total_rank_decisions=0;max_length_difference=0.
    seals={(r['fold'],r['shot']):r for r in READ(root/'prediction_receipts.json')}
    for plan in plans:
        f=plan['fold'];folder=root/f'fold{f}';access=FoldAccess(plan,source_ids)
        def signal_guard(device,sid,**kwargs):
            access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
        def outcome_guard(device,sid):
            access.check('outcome',device,sid);return read_outcome(device,sid)
        saved=torch.load(folder/'parent.pt',map_location='cuda',weights_only=True)
        model=copy.deepcopy(parent);model.load_state_dict(saved['model']);model.set_stage('frozen')
        allowed=tuple(model.encoder_names())+('base.forecast_head.weight','base.forecast_head.bias')
        assert_unchanged(model,original,allowed);frozen=state_cpu(model)
        assert sha(folder/'parent.pt')==READ(folder/'preparation_receipt.json')['checkpoint_sha256']
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
            records=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in plan['fit']]
            stats=fit_statistics([{k:r[k] for k in SIGNAL_KEYS} for r in records],device='EAST',allowed_ids=plan['fit'])
            mapping=make_map(source_stats,stats,'density_gain');assert mapping==READ(folder/'input_map.json')
            target=[dict(r,**transform({k:r[k] for k in ('values','available')},mapping)) for r in records]
            training_counts[f'fold{f}']=check_manifests(folder,source,target,set(plan['fit']))
            memory,manifest=make_memory(model,target)
            assert manifest==READ(folder/'memory_manifest.json')
            cached=torch.load(folder/'memory.pt',map_location='cpu',weights_only=True)
            for k,v in memory.items():torch.testing.assert_close(torch.as_tensor(v).cpu(),cached[k],rtol=0,atol=0)
            access.phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                scores,updates=predict(model,signal,memory)
                up=folder/'shots'/f'{sid}_updates.json';path=folder/'shots'/f'{sid}.npz'
                assert sha(path)==seals[(f,sid)]['prediction_sha256'] and sha(up)==seals[(f,sid)]['updates_sha256']
                assert updates==READ(up)
                with np.load(path,allow_pickle=False) as trace:
                    np.testing.assert_array_equal(trace['times'],s['times'])
                    for a in ARMS:
                        np.testing.assert_array_equal(scores[a],trace[a])
                        np.testing.assert_array_equal(trace[a][:256],trace['static'][:256])
                for a,rows in updates.items():
                    for j,r in enumerate(rows):
                        assert r['last_observation']==255+128*j and r['first_affected_prediction']==256+128*j
                        assert r['first_affected_prediction']<len(s['times'])
                        assert r['parameter_delta_norm']<=.00101
                        update_stats[a]['steps']+=r['gradient_steps']
                        update_stats[a]['loss_decreased']+=int(r.get('loss_after',float('inf'))<r.get('loss_before',-float('inf')))
                        if a in RANK_ARMS:
                            assert set(r['selected_shots'])<=set(plan['fit']) and r['selected_labels']==[0,0,1,1]
                            assert r['selected_shots']==memory['shots'][r['selected_indices']].tolist()
                for j,r in enumerate(updates['now_rank']):
                    total_rank_decisions+=1;lengths=[updates[a][j]['parameter_delta_norm'] for a in RANK_ARMS]
                    max_length_difference=max(max_length_difference,max(lengths)-min(lengths))
                    for a in RANK_ARMS:
                        selections[a]+=int(set(updates[a][j]['selected_indices'])!=set(r['selected_indices']))
                replay.append(dict(fold=f,shot=sid,arms=len(ARMS),exact=True))
                print(dict(stage='replay',fold=f,shot=sid,done=len(replay)),flush=True)
            assert_unchanged(model,frozen)
        # Original run access log must match exactly15 training signal/labels,
        # then five held signals. The audit has the same read schedule.
        assert access.log==READ(folder/'access.json')
        for sid in plan['held']:
            # Scoring reads are deliberately outside the prediction boundary.
            # They cannot reach the already-computed scores or fast states.
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            with np.load(folder/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                valid=s['valid']
                for a in ARMS:
                    r=independently_score(sid,s['is_disrupt'],s['down'],s['times'][valid],trace[a][valid],p['threshold'])
                    summaries[a].append(r)
                    details.append(dict(fold=f,arm=a,**r,max_valid_probability=float(trace[a][valid].max())))
        del model,memory
    results={a:aggregate(rows,p['threshold']) for a,rows in summaries.items()}
    expected=READ(root/'per_shot.json')
    assert results==done['metrics'] and gate(results)==done['recommend_test360']
    for a in ARMS:assert summaries[a]==expected[a]['predictions']
    assert sum(s['steps'] for s in update_stats.values())==done['neural_steps'] and max_length_difference<2e-7
    # Refit fold0 from the original J-TEXT initialization, using only its15 shots.
    out=root/'audit_refit0';out.mkdir();plan=plans[0];access=FoldAccess(plan,source_ids)
    def signal_guard(device,sid,**kwargs):
        access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
    def outcome_guard(device,sid):
        access.check('outcome',device,sid);return read_outcome(device,sid)
    with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
        refit,mapping,_=prepare(parent,plan['fit'],source,source_stats,out)
    original_fold=torch.load(root/'fold0'/'parent.pt',map_location='cpu',weights_only=True)['model']
    for k,v in refit.state_dict().items():torch.testing.assert_close(v.cpu(),original_fold[k],rtol=0,atol=0)
    for name in ('input_map.json','risk_training_manifest.jsonl','forecast_manifest.jsonl','risk_training.json','forecast_training.json'):
        assert (out/name).read_bytes()==(root/'fold0'/name).read_bytes(),name
    assert_unchanged(parent,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'diagnostic_details.json',dict(rows=details,updates=update_stats,
        changed_selected_sets_vs_now=selections,rank_decisions=total_rank_decisions))
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        audit_test_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_horizon_memory_audit.py'),
        diagnostic_sha256=sha(root/'diagnostic_details.json'),independent_shot_recounts=len(details),
        exact_cuda_replayed_shots=len(replay),arms_per_shot=len(ARMS),replay=replay,
        exact_refitted_folds=[0],training_draws_checked=training_counts,
        source_and_fold_frozen_state_verified=True,fold_input_maps_rebuilt=True,memory_rebuilt=True,
        max_step_length_difference=max_length_difference,metrics=results,gate=gate(results),
        caveat='same implementation neural replay; four disruptive shots; development cross-validation, not independent future TEST')
    _durable_json(root/'verification.json',report)
    print(dict(status='VERIFIED',shots=len(replay),recounts=len(details),gate=gate(results)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
