"""Paired 2x2 preparation / static-versus-intra whole-shot development study."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..horizon_memory_20261023.learning import FoldAccess
from ..horizon_memory_20261023.run import snapshot,SELECTION
from ..learned_update_loss_20261023.learning import folds
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import fit_statistics,transform,SIGNAL_KEYS
from ..warmstart_alignment_20261022.run import public
from .learning import CONDITIONS,predict
from .prepare import prepare

ROOT=io23.ROOT
PLAN=ROOT/'docs/research/source_event_rebuild_20261022/输入映射与表示对齐_四组对照事前约定_20261023.md'
REFERENCE=ROOT/'experiments/risk_aligned_online/runs/horizon_memory_cv20_20261023'
REFERENCE_SHA='3757f0d3e37c1093304519d1c1778fcd3582b033523aafb7958919eb506420c4'
READ=lambda p:json.loads(p.read_text())


def score(root,plans,threshold):
    rows={f'{c}_{a}':[] for c in CONDITIONS for a in ('static','intra')};details=[];access=[]
    for plan in plans:
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            access.append(dict(fold=plan['fold'],shot=sid,phase='score_after_all_seals'))
            valid=s['valid'];timely=valid & (np.round(1000*(s['down']-s['times']),9)>30.) if s['is_disrupt'] else valid
            for c in CONDITIONS:
                with np.load(root/f'fold{plan["fold"]}'/c/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                    np.testing.assert_array_equal(trace['times'],s['times'])
                    for a in ('static','intra'):
                        name=f'{c}_{a}'
                        rows[name].append(dict(shot=sid,times=s['times'][valid],scores=trace[a][valid],is_disrupt=s['is_disrupt'],down=s['down']))
                        details.append(dict(fold=plan['fold'],condition=c,mode=a,shot=sid,is_disrupt=s['is_disrupt'],
                            max_valid=float(trace[a][valid].max()),max_timely=float(trace[a][timely].max()) if timely.any() else None))
    results={a:metrics(v,threshold,lead=30.) for a,v in rows.items()}
    _durable_json(root/'evaluation_access.json',access);_durable_json(root/'per_shot.json',results)
    _durable_json(root/'risk_peaks.json',details)
    return results


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    assert sha(REFERENCE/'completed.json')==REFERENCE_SHA
    old=READ(REFERENCE/'completed.json');old_protocol=READ(REFERENCE/'protocol.json')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    plans=folds(READ(SELECTION)['selected']);assert plans==old_protocol['folds']
    base,info=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda();parent.set_stage('frozen')
    original=state_cpu(parent);threshold=float(info['threshold']);source_ids=io23.source_roles()['train']
    root.mkdir(parents=True)
    protocol=dict(source_checkpoint=str(DEFAULT_PARENT),source_sha256=PARENT_SHA256,selection_sha256=sha(SELECTION),
        reference=str(REFERENCE),reference_completed_sha256=REFERENCE_SHA,plan_path=str(PLAN),plan_sha256=sha(PLAN),
        code_sha256=snapshot(root),test_sha256=sha(ROOT/'experiments/risk_aligned_online/tests/test_preparation_factorial.py'),
        folds=plans,conditions=CONDITIONS,threshold=threshold,seed=131,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),scope='development CV20, NOT TEST360 or chronological independent future',
        risk_steps=512,forecast_steps=256,matched_draws=True,source_training_role_ids=source_ids,
        no_inter=True,no_alarm_adjustment=True,independent_confirmation=False)
    _durable_json(root/'protocol.json',protocol)
    read_outcome,read_signal=io23.read_outcome,simple_data.read_shot;source_access=[]
    def source_guard(device,sid):
        if device!='J-TEXT' or sid not in source_ids:raise AssertionError('source TRAIN only')
        source_access.append(sid);return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    _durable_json(root/'source_label_access.json',source_access)
    source_stats=fit_statistics([{k:s[k] for k in SIGNAL_KEYS} for s in source],device='J-TEXT',allowed_ids=source_ids)
    _durable_json(root/'source_statistics.json',source_stats)
    seals=[];steps=0;matched=[];exact_reference=[]
    for plan in plans:
        f=plan['fold'];fold=root/f'fold{f}';fold.mkdir()
        for condition in CONDITIONS:
            out=fold/condition;out.mkdir();(out/'shots').mkdir();access=FoldAccess(plan,source_ids)
            def signal_guard(device,sid,**kwargs):
                access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
            def outcome_guard(device,sid):
                access.check('outcome',device,sid);return read_outcome(device,sid)
            with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
                model,mapping,_=prepare(parent,plan['fit'],source,source_stats,out,condition)
                for name in ('risk_training_manifest.jsonl','forecast_manifest.jsonl'):
                    if condition!=CONDITIONS[0]:assert (out/name).read_bytes()==(fold/CONDITIONS[0]/name).read_bytes()
                matched.append(dict(fold=f,condition=condition,draws_equal=True))
                if condition=='mapped_moments':
                    ref=REFERENCE/f'fold{f}';path=ref/'parent.pt'
                    assert sha(path)==old['artifact_sha256'][str(path.relative_to(REFERENCE))]
                    checkpoint=torch.load(path,map_location='cpu',weights_only=True)['model']
                    for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),checkpoint[k],rtol=0,atol=0)
                    for name in ('input_map.json','risk_training_manifest.jsonl','forecast_manifest.jsonl','risk_training.json','forecast_training.json'):
                        assert (out/name).read_bytes()==(ref/name).read_bytes(),name
                frozen=state_cpu(model);access.phase='predict'
                for sid in plan['held']:
                    s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                    signal=transform({k:s[k] for k in ('values','available')},mapping)
                    scores,updates=predict(model,signal)
                    path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
                    np.savez_compressed(path,times=s['times'],**scores);_durable_json(up,updates)
                    steps+=sum(r['gradient_steps'] for r in updates)
                    if condition=='mapped_moments':
                        refpath=REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz'
                        assert sha(refpath)==old['artifact_sha256'][str(refpath.relative_to(REFERENCE))]
                        with np.load(refpath,allow_pickle=False) as trace:
                            np.testing.assert_array_equal(scores['static'],trace['static']);np.testing.assert_array_equal(scores['intra'],trace['forecast'])
                        assert updates==READ(refpath.with_name(f'{sid}_updates.json'))['forecast']
                        exact_reference.append(dict(fold=f,shot=sid,exact=True))
                    seals.append(dict(fold=f,condition=condition,shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
                assert_unchanged(model,frozen)
            _durable_json(out/'access.json',access.log);assert_unchanged(parent,original)
            print(dict(stage='condition_complete',fold=f,condition=condition,shots=len(seals),seconds=time.monotonic()-started),flush=True)
            del model
    _durable_json(root/'prediction_receipts.json',seals)
    results=score(root,plans,threshold)
    summary={a:dict(**public(r),very_early_over3000=sum(p['timely'] and p['lead_ms']>3000 for p in r['predictions'])) for a,r in results.items()}
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics=summary,seconds=time.monotonic()-started,neural_steps=steps,
        matched_draws=matched,exact_old_reference=exact_reference,source_unchanged=True,test360_accessed=False,
        independent_confirmation=False,artifact_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
    _durable_json(root/'completed.json',done)
    print(json.dumps({k:v for k,v in done.items() if k not in ('artifact_sha256','matched_draws','exact_old_reference')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
