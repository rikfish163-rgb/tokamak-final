"""2x2 history experiment: known-class selection versus source supervision."""
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
from ..temporal_scope_20261023.run import REFERENCE,REF_SHA
from .design import class_aware_selection
from .learning import prepare

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/准备类别覆盖与源监督_四组事前约定_20261023.md'
SHOTLIST=[io23.ROOT/'shot_list/EAST'/name for name in ('Disruption_EAST_TV.json','Non-disruption_EAST_TV.json')]
METHODS=('blind_mixed','blind_target','aware_mixed','aware_target')
NEW=METHODS[1:]
ARMS=('source',)+tuple(m+'_'+s for m in METHODS for s in ('static','intra'))


def score(root,plans,threshold):
    all_rows={a:[] for a in ARMS};folds={};access=[]
    for plan in plans:
        rows={a:[] for a in ARMS}
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            access.append(dict(fold=plan['fold'],shot=sid,phase='after_all_prediction_seals'))
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(z['times'],s['times'])
                for a in ARMS:
                    r=dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down'])
                    rows[a].append(r);all_rows[a].append(r)
        folds[str(plan['fold'])]={a:public(metrics(v,threshold,lead=30.)) for a,v in rows.items()}
    results={a:metrics(v,threshold,lead=30.) for a,v in all_rows.items()}
    _durable_json(root/'per_shot.json',results);_durable_json(root/'evaluation_access.json',access)
    return {a:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for a,v in results.items()},folds


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    assert sha(REFERENCE/'completed.json')==REF_SHA
    reference=READ(REFERENCE/'completed.json');rp=READ(REFERENCE/'protocol.json')
    for path,h in reference['artifact_sha256'].items():assert sha(REFERENCE/path)==h
    assert not set(rp['shot_ids'])&set(io23.cohort_ids('EAST'))
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    parent,threshold=make_parent();origin=state_cpu(parent);root.mkdir(parents=True)
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_preparation_factors.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(folds=rp['folds'],shot_ids=rp['shot_ids'],threshold=threshold,reference_completed_sha256=REF_SHA,
        source_sha256=PARENT_SHA256,plan_sha256=sha(PLAN),code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in files},
        input_sha256=rp['input_sha256'],shotlist_sha256={str(v):sha(v) for v in SHOTLIST},
        seed=131,risk_steps=512,forecast_steps=256,arms=ARMS,
        information_contract='aware reads candidate shotlist classes; NOT equal blind-query budget; 20 gradient identities/model',
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        independent_confirmation=False,test360_accessed=False,inter_steps=0,alarm_updates=0)
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for v in sorted(files):z.write(v,str(v.relative_to(io23.ROOT)))
    disruptive,safe=[set(READ(path)) for path in SHOTLIST];selections={}
    for plan in p['folds']:
        f=plan['fold'];out=root/f'fold{f}';out.mkdir();(out/'shots').mkdir()
        with np.load(REFERENCE/f'fold{f}'/'selection_features.npz') as z:
            ids=z['ids'].tolist();x=z['descriptors'].copy()
        assert ids==plan['candidate'] and all((s in disruptive)!=(s in safe) for s in ids)
        labels={s:s in disruptive for s in ids}
        blind=sorted(READ(REFERENCE/f'fold{f}'/'selection.json')['selected']['diverse'])
        aware=class_aware_selection(ids,x,labels);selections[f]=dict(blind=blind,aware=aware)
        _durable_json(out/'selection.json',dict(selected=selections[f],candidate_class_access=labels,
            source_descriptor_sha256=sha(REFERENCE/f'fold{f}'/'selection_features.npz')))
    _durable_json(root/'selection_seals.json',{str(f):sha(root/f'fold{f}'/'selection.json') for f in selections})
    reader=io23.read_outcome;source_ids=set(io23.source_roles()['train'])
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    access=[];receipts=[];steps=0
    for plan in p['folds']:
        f=plan['fold'];out=root/f'fold{f}';models={}
        for method in NEW:
            selected=selections[f]['blind' if method.startswith('blind') else 'aware'];folder=out/method;folder.mkdir()
            def train_guard(device,sid):
                assert device=='EAST' and sid in selected and sid not in plan['held']
                access.append(dict(fold=f,method=method,shot=sid,phase='selected_preparation'))
                return reader(device,sid)
            with patch.object(io23,'read_outcome',side_effect=train_guard):
                target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in selected]
                assert all(s['is_disrupt']==(s['shot'] in disruptive) for s in target)
                model=copy.deepcopy(parent);prepare(model,source,target,folder,use_source=method.endswith('mixed'));models[method]=model
            matched=REFERENCE/f'fold{f}'/'diverse' if method=='blind_target' else out/'aware_mixed'
            if method!='aware_mixed':
                for name in ('risk_manifest.jsonl','forecast_manifest.jsonl'):assert (folder/name).read_bytes()==(matched/name).read_bytes()
        states={m:state_cpu(model) for m,model in models.items()}
        def deny(*args,**kwargs):raise AssertionError('held outcomes forbidden during prediction')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'));signal={k:s[k] for k in ('values','available')}
                curves={};updates={}
                for method,model in models.items():
                    scores,u,_=predict(model,signal);updates[method]=u
                    np.testing.assert_array_equal(scores['static'][:256],scores['intra'][:256])
                    curves.update({method+'_'+a:v for a,v in scores.items()});steps+=sum(r['gradient_steps'] for r in u)
                with np.load(REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz') as z:
                    np.testing.assert_array_equal(z['times'],s['times'])
                    curves.update(source=z['source'].copy(),blind_mixed_static=z['diverse_static'].copy(),blind_mixed_intra=z['diverse_intra'].copy())
                path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
                np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
                receipts.append(dict(fold=f,shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
                if len(receipts)%5==0:print(dict(stage='predict',shots=len(receipts),seconds=time.monotonic()-start),flush=True)
        for m,model in models.items():assert_unchanged(model,states[m])
        assert_unchanged(parent,origin)
    _durable_json(root/'prediction_receipts.json',receipts);_durable_json(root/'training_label_access.json',access)
    results,folds=score(root,p['folds'],threshold)
    for a,b in (('source','source'),('blind_mixed_static','diverse_static'),('blind_mixed_intra','diverse_intra')):assert results[a]==reference['metrics'][b]
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics=results,fold_metrics=folds,seconds=time.monotonic()-start,
        offline_steps=4*3*768,online_steps=steps,source_unchanged=True,test360_accessed=False,independent_confirmation=False,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
