"""Four-fold historical development experiment with a label-blind budget20."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..increment_forecast_20261023.learning import IncrementFeedback,predict
from ..redesign_20261004.prepare_prior import load_parent,load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.run import predict_many,public
from .selection import describe,select,identity_folds,rank_coordinates
from .learning import prepare

READ=lambda p:json.loads(p.read_text())
PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/support_geometry85_20261023'
PREVIOUS_SHA='f94c625def623f3c0ebfefbb8e6ece9fbd82dba19a48ec475be98b28ac626d36'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/等预算信号多样性准备_事前约定_20261023.md'
METHODS=('uniform','diverse')
ARMS=('source','uniform_static','uniform_intra','diverse_static','diverse_intra')


def make_parent():
    base,info=load_parent(DEFAULT_PARENT)
    model=IncrementFeedback(base,'increment').cuda();model.set_stage('frozen')
    return model,float(info['threshold'])


def coverage(x,ids,selected):
    z=rank_coordinates(x);idx=[ids.index(s) for s in selected]
    d=np.sqrt(((z[:,None]-z[idx][None])**2).sum(-1)).min(1)
    return dict(max_distance=float(d.max()),mean_distance=float(d.mean()))


def score(root,plans,threshold):
    rows={a:[] for a in ARMS};fold_metrics={};access=[]
    for plan in plans:
        local={a:[] for a in ARMS}
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            access.append(dict(fold=plan['fold'],shot=sid,phase='after_all_seals'))
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(z['times'],s['times'])
                for a in ARMS:
                    r=dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down'])
                    rows[a].append(r);local[a].append(r)
        fold_metrics[str(plan['fold'])]={a:public(metrics(v,threshold,lead=30.)) for a,v in local.items()}
    results={a:metrics(v,threshold,lead=30.) for a,v in rows.items()}
    _durable_json(root/'per_shot.json',results);_durable_json(root/'evaluation_access.json',access)
    flips={}
    for method in METHODS:
        before=results[method+'_static']['predictions'];after=results[method+'_intra']['predictions']
        changed=[]
        for a,b in zip(before,after):
            assert a['shot']==b['shot']
            good_a=a['timely'] if a['is_disrupt'] else not a['alarm']
            good_b=b['timely'] if b['is_disrupt'] else not b['alarm']
            if good_a!=good_b:changed.append(dict(shot=a['shot'],is_disrupt=a['is_disrupt'],beneficial=good_b,before=a,after=b))
        flips[method]=changed
    _durable_json(root/'flips.json',flips)
    return {a:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for a,v in results.items()},fold_metrics


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    assert sha(PREVIOUS/'completed.json')==PREVIOUS_SHA
    old=READ(PREVIOUS/'protocol.json');ids=old['shot_ids'];plans=identity_folds(ids)
    assert not set(ids)&set(io23.cohort_ids('EAST'))
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    parent,threshold=make_parent();original=state_cpu(parent)
    center=parent.base.center.detach().cpu().numpy();scale=parent.base.scale.detach().cpu().numpy()
    root.mkdir(parents=True)
    paths=set(Path(__file__).parent.glob('*.py'));paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_diverse_support.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    protocol=dict(folds=plans,shot_ids=ids,threshold=threshold,source_sha256=PARENT_SHA256,
        previous_completed_sha256=PREVIOUS_SHA,plan_sha256=sha(PLAN),code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in paths},
        input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in ids},
        methods=METHODS,arms=ARMS,label_budget_per_fold_and_method=20,
        input_contract='flat23,source asinh,no empirical target gain,CIII sentinel mask,EAST MNM unavailable',
        seed=131,risk_steps=512,forecast_steps=256,inter_steps=0,alarm_updates=0,
        test360_accessed=False,independent_confirmation=False,scope='historical85 development,not chronological first20',
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for v in sorted(paths):z.write(v,str(v.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('outcome access forbidden in signal-only phase')
    selections={}
    # Seal all acquisition rules BEFORE retrieving any outcome in this run.
    with patch.object(io23,'read_outcome',side_effect=deny):
        for plan in plans:
            f=plan['fold'];out=root/f'fold{f}';out.mkdir();(out/'shots').mkdir()
            desc=[]
            for sid in plan['candidate']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                desc.append(describe({k:s[k] for k in ('values','available')},center,scale))
            x=np.stack(desc);np.savez_compressed(out/'selection_features.npz',ids=np.asarray(plan['candidate']),descriptors=x)
            selections[f]={m:select(plan['candidate'],x,20,m,131+f) for m in METHODS}
            _durable_json(out/'selection.json',dict(selected=selections[f],
                coverage={m:coverage(x,plan['candidate'],selections[f][m]) for m in METHODS},outcomes_accessed=0))
    _durable_json(root/'selection_seals.json',{str(f):sha(root/f'fold{f}'/'selection.json') for f in selections})
    reader=io23.read_outcome;source_ids=io23.source_roles()['train'];label_access=[]
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    receipts=[];online_steps=0
    for plan in plans:
        f=plan['fold'];out=root/f'fold{f}';models={}
        for method in METHODS:
            selected=sorted(selections[f][method]);folder=out/method;folder.mkdir()
            def training_guard(device,sid):
                assert device=='EAST' and sid in selected and sid not in plan['held']
                label_access.append(dict(fold=f,method=method,shot=int(sid),phase='after_selection_before_fit'))
                return reader(device,sid)
            with patch.object(io23,'read_outcome',side_effect=training_guard):
                target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in selected]
                model=copy.deepcopy(parent);prepare(model,source,target,folder);models[method]=model
            print(dict(stage='prepared',fold=f,method=method,n_disrupt=sum(s['is_disrupt'] for s in target),seconds=time.monotonic()-start),flush=True)
        states={m:state_cpu(model) for m,model in models.items()}
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal={k:s[k] for k in ('values','available')};curves=predict_many(parent,dict(source=signal),device='cuda');updates={}
                for method,model in models.items():
                    scores,u,_=predict(model,signal)
                    np.testing.assert_array_equal(scores['static'][:256],scores['intra'][:256])
                    curves.update({method+'_'+k:v for k,v in scores.items()});updates[method]=u
                    online_steps+=sum(r['gradient_steps'] for r in u)
                path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
                np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
                receipts.append(dict(fold=f,shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
                if len(receipts)%5==0:print(dict(stage='prediction',sealed=len(receipts),seconds=time.monotonic()-start),flush=True)
        for m,model in models.items():assert_unchanged(model,states[m])
        assert_unchanged(parent,original)
    _durable_json(root/'prediction_receipts.json',receipts);_durable_json(root/'training_label_access.json',label_access)
    summary,folds=score(root,plans,threshold)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='COMPLETED',metrics=summary,fold_metrics=folds,seconds=time.monotonic()-start,
        online_steps=online_steps,offline_steps=4*2*(512+256),source_unchanged=True,
        test360_accessed=False,independent_confirmation=False,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',result);print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
