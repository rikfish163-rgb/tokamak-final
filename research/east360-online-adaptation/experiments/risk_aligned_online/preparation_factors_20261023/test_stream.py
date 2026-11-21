"""Fixed historical20 preparation, followed by all official TEST360 identities.

This is development-exposed test replay, not independent confirmation and not
the first20 chronological shots. The class-aware rule observes history labels.
"""
import argparse
import copy
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
from ..diverse_support_20261023.run import make_parent,READ
from ..diverse_support_20261023.selection import describe,select
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.run import predict_many
from .design import class_aware_selection
from .learning import prepare
from .run import METHODS,ARMS,SHOTLIST,score

HISTORY=io23.ROOT/'experiments/risk_aligned_online/runs/preparation_factors85_20261023'
HISTORY_SHA='9180eff55301a9433cde3e5fcc1ea347c5d5f68e58aa2a6e5d953af5cc333af7'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/历史20炮准备_TEST360固定评价约定_20261023.md'


def select_preparation(ids,descriptors,labels,test_ids):
    if (len(ids)!=85 or len(set(ids))!=85 or len(test_ids)!=360
            or len(set(test_ids))!=360 or set(ids)&set(test_ids)):
        raise ValueError('distinct historical85 and disjoint TEST360 required')
    aware=class_aware_selection(ids,descriptors,labels)
    blind=sorted(select(ids,descriptors,20,'diverse',131))
    return dict(blind=blind,aware=aware)


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    assert sha(HISTORY/'completed.json')==HISTORY_SHA
    hp=READ(HISTORY/'protocol.json');ids=hp['shot_ids'];test_ids=io23.cohort_ids('EAST')
    assert not set(ids)&set(test_ids)
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    parent,threshold=make_parent();origin=state_cpu(parent)
    root.mkdir(parents=True);out=root/'fold0';out.mkdir();(out/'shots').mkdir()
    # One fixed preparation/test partition; fold0 is a file-layout reuse, not CV.
    plans=[dict(fold=0,candidate=ids,held=test_ids)]
    files=set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_preparation_test360.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(folds=plans,shot_ids=test_ids,history_ids=ids,threshold=threshold,arms=ARMS,
        history_completed_sha256=HISTORY_SHA,source_sha256=PARENT_SHA256,plan_sha256=sha(PLAN),
        code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in files},
        input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in ids+test_ids},
        shotlist_sha256={str(v):sha(v) for v in SHOTLIST},
        input_contract='flat23,source asinh,no empirical gain,CIII sentinel mask,EAST MNM unavailable',
        information_contract='candidate signal access85; aware also existing classes85; gradient fit20/model',
        seed=131,risk_steps=512,forecast_steps=256,inter_steps=0,alarm_updates=0,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        test360_accessed=True,independent_confirmation=False,
        scope='official TEST360 development replay; NOT chronological first20 or independent future')
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for v in sorted(files):z.write(v,str(v.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('outcomes forbidden before prediction seal')
    center=parent.base.center.detach().cpu().numpy();scale=parent.base.scale.detach().cpu().numpy()
    with patch.object(io23,'read_outcome',side_effect=deny):
        descriptors=[]
        for sid in ids:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            descriptors.append(describe({k:s[k] for k in ('values','available')},center,scale))
    x=np.stack(descriptors);disruptive,safe=[set(READ(v)) for v in SHOTLIST]
    assert all((s in disruptive)!=(s in safe) for s in ids)
    labels={s:s in disruptive for s in ids};selected=select_preparation(ids,x,labels,test_ids)
    np.savez_compressed(out/'selection_features.npz',ids=np.asarray(ids),descriptors=x)
    _durable_json(out/'selection.json',dict(selected=selected,candidate_class_access=labels))
    _durable_json(root/'selection_seal.json',dict(sha256=sha(out/'selection.json')))
    print(dict(stage='selection',ids=selected,disruptive={k:sum(labels[s] for s in v) for k,v in selected.items()}),flush=True)
    reader=io23.read_outcome;source_ids=set(io23.source_roles()['train'])
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    models={};access=[]
    for method in METHODS:
        fit=selected[method.split('_')[0]];folder=out/method;folder.mkdir()
        def train_guard(device,sid):
            assert device=='EAST' and sid in fit and sid not in test_ids
            access.append(dict(method=method,shot=sid,phase='historical_preparation'))
            return reader(device,sid)
        with patch.object(io23,'read_outcome',side_effect=train_guard):
            target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in fit]
            assert all(s['is_disrupt']==labels[s['shot']] for s in target)
            model=copy.deepcopy(parent);prepare(model,source,target,folder,use_source=method.endswith('mixed'));models[method]=model
        if method.endswith('target'):
            other=out/(method.split('_')[0]+'_mixed')
            for name in ('risk_manifest.jsonl','forecast_manifest.jsonl'):assert (folder/name).read_bytes()==(other/name).read_bytes()
    _durable_json(root/'training_label_access.json',access)
    _durable_json(root/'model_seals.json',{m:sha(out/m/'prepared.pt') for m in METHODS})
    frozen={m:state_cpu(v) for m,v in models.items()};receipts=[];steps=0
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in test_ids:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal={k:s[k] for k in ('values','available')};curves=predict_many(parent,dict(source=signal),device='cuda');updates={}
            for method,model in models.items():
                scores,u,_=predict(model,signal);np.testing.assert_array_equal(scores['static'][:256],scores['intra'][:256])
                curves.update({method+'_'+a:v for a,v in scores.items()});updates[method]=u;steps+=sum(v['gradient_steps'] for v in u)
            path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
            receipts.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
            if len(receipts)%10==0:print(dict(stage='predict',shots=len(receipts),seconds=time.monotonic()-start),flush=True)
    for m,v in models.items():assert_unchanged(v,frozen[m])
    assert_unchanged(parent,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'prediction_receipts.json',receipts)
    results,_=score(root,plans,threshold)
    assert all(v['n_disrupt']==180 and v['n_safe']==180 for v in results.values())
    done=dict(status='COMPLETED',metrics=results,seconds=time.monotonic()-start,offline_steps=4*768,
        online_steps=steps,source_unchanged=True,independent_confirmation=False,test360_accessed=True,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
