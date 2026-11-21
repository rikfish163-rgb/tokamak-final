"""Matched historical20 joint classifier experiment; exposed TEST360 replay."""
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
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.run import public
from .learning import prepare

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/preparation_factors20_test360_20261023'
REF_SHA='0002645f9b65d1dd975da639152dd2c2867c6bd95e614a51942e7dfdf4ddc7ef'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/既有分类网络联合适配_事前约定_20261023.md'
METHODS=('first_head','full_classifier')
ARMS=('source','first_fixed_head_static','first_fixed_head_intra')+tuple(m+'_'+a for m in METHODS for a in ('static','intra'))
REUSED={'source':'source','first_fixed_head_static':'blind_mixed_static','first_fixed_head_intra':'blind_mixed_intra'}


def score(root,p):
    rows={a:[] for a in ARMS};access=[]
    for sid in p['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        access.append(dict(shot=sid,phase='after_all_prediction_seals'))
        with np.load(root/'fold0/shots'/f'{sid}.npz') as z:
            np.testing.assert_array_equal(z['times'],s['times'])
            for a in ARMS:
                rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    results={a:metrics(v,p['threshold'],lead=30.) for a,v in rows.items()}
    _durable_json(root/'per_shot.json',results);_durable_json(root/'evaluation_access.json',access)
    return {a:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for a,v in results.items()}


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    assert sha(REFERENCE/'completed.json')==REF_SHA
    ref=READ(REFERENCE/'completed.json');rp=READ(REFERENCE/'protocol.json')
    for name,h in ref['artifact_sha256'].items():assert sha(REFERENCE/name)==h,name
    selected=READ(REFERENCE/'fold0/selection.json')['selected']['blind'];held=io23.cohort_ids('EAST')
    assert held==rp['shot_ids'] and len(selected)==20 and not set(selected)&set(held)
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    parent,threshold=make_parent();origin=state_cpu(parent);assert threshold==rp['threshold']
    root.mkdir(parents=True);folder=root/'fold0';folder.mkdir();(folder/'shots').mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_joint_classifier.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(folds=[dict(fold=0,candidate=rp['history_ids'],held=held)],shot_ids=held,selected=selected,
        threshold=threshold,arms=ARMS,reference_completed_sha256=REF_SHA,source_sha256=PARENT_SHA256,
        plan_sha256=sha(PLAN),code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in files},
        input_sha256=rp['input_sha256'],shotlist_sha256=rp['shotlist_sha256'],
        input_contract=rp['input_contract'],seed=131,risk_steps=512,forecast_steps=256,
        information_contract='fixed previous blind20; original candidate signal access85; selected after prior development results',
        scope='historical20 preparation and development-exposed TEST360; NOT chronological first20',
        independent_confirmation=False,test360_accessed=True,inter_steps=0,alarm_updates=0,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p);_durable_json(folder/'selection.json',dict(selected=selected))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for v in sorted(files):z.write(v,str(v.relative_to(io23.ROOT)))
    reader=io23.read_outcome;source_ids=set(io23.source_roles()['train'])
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    models={};access=[]
    for method in METHODS:
        out=folder/method;out.mkdir()
        def train_guard(device,sid):
            assert device=='EAST' and sid in selected and sid not in held
            access.append(dict(method=method,shot=sid,phase='historical_preparation'))
            return reader(device,sid)
        with patch.object(io23,'read_outcome',side_effect=train_guard):
            target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in selected]
            assert sum(s['is_disrupt'] for s in target)==4
            model=copy.deepcopy(parent);prepare(model,source,target,out,scope=method);models[method]=model
        for name in ('risk_manifest.jsonl','forecast_manifest.jsonl'):
            assert (out/name).read_bytes()==(REFERENCE/'fold0/blind_mixed'/name).read_bytes(),name
    _durable_json(root/'training_label_access.json',access)
    _durable_json(root/'model_seals.json',{m:sha(folder/m/'prepared.pt') for m in METHODS})
    frozen={m:state_cpu(v) for m,v in models.items()};receipts=[];steps=0
    def deny(*args,**kwargs):raise AssertionError('no test outcomes before all prediction seals')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in held:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal={k:s[k] for k in ('values','available')};curves={};updates={}
            for method,model in models.items():
                scores,u,_=predict(model,signal);np.testing.assert_array_equal(scores['static'][:256],scores['intra'][:256])
                curves.update({method+'_'+a:v for a,v in scores.items()});updates[method]=u;steps+=sum(v['gradient_steps'] for v in u)
            with np.load(REFERENCE/'fold0/shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(z['times'],s['times'])
                curves.update({a:z[b].copy() for a,b in REUSED.items()})
            path=folder/'shots'/f'{sid}.npz';up=folder/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
            receipts.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
            if len(receipts)%10==0:print(dict(stage='predict',shots=len(receipts),seconds=time.monotonic()-start),flush=True)
    for m,v in models.items():assert_unchanged(v,frozen[m])
    assert_unchanged(parent,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'prediction_receipts.json',receipts);results=score(root,p)
    assert all(v['n_disrupt']==180 and v['n_safe']==180 for v in results.values())
    for a,b in REUSED.items():assert results[a]==ref['metrics'][b]
    done=dict(status='COMPLETED',metrics=results,seconds=time.monotonic()-start,offline_steps=2*768,
        online_steps=steps,source_unchanged=True,independent_confirmation=False,test360_accessed=True,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
