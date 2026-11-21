"""Same acquired20 and losses: allow the second temporal block to learn EAST."""
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
from ..diverse_support_20261023.learning import prepare
from ..diverse_support_20261023.run import make_parent,READ
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..simple_train import metrics,sha
from ..simple_online import _durable_json
from ..warmstart_alignment_20261022.run import public
from .model import BothBlockPreparation,to_deployment

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/diverse_support85_20261023'
REF_SHA='d24cf7d40ccc819ac98b159efbe14c402b001175e1539e8885b397f3c82d3286'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/第二时序层准备_事前约定_20261023.md'
ARMS=('source','first_static','first_intra','both_static','both_intra')


def score(root,plans,threshold):
    all_rows={a:[] for a in ARMS};folds={};access=[]
    for plan in plans:
        rows={a:[] for a in ARMS}
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            access.append(dict(fold=plan['fold'],shot=sid,phase='after_all_seals'))
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
    assert sha(REFERENCE/'completed.json')==REF_SHA
    reference=READ(REFERENCE/'completed.json');rp=READ(REFERENCE/'protocol.json')
    for name,h in reference['artifact_sha256'].items():assert sha(REFERENCE/name)==h
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    parent,threshold=make_parent();origin=state_cpu(parent);root.mkdir(parents=True)
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_temporal_scope.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(folds=rp['folds'],shot_ids=rp['shot_ids'],threshold=threshold,reference_completed_sha256=REF_SHA,
        source_sha256=PARENT_SHA256,plan_sha256=sha(PLAN),code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in files},
        input_sha256=rp['input_sha256'],risk_steps=512,forecast_steps=256,
        sole_change='extra second Transformer block in EAST offline preparation; intra remains first-only',
        independent_confirmation=False,test360_accessed=False)
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for v in sorted(files):z.write(v,str(v.relative_to(io23.ROOT)))
    reader=io23.read_outcome;source_ids=io23.source_roles()['train']
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    access=[];receipts=[];steps=0
    for plan in p['folds']:
        f=plan['fold'];ref=REFERENCE/f'fold{f}';out=root/f'fold{f}';out.mkdir();(out/'shots').mkdir()
        selected=sorted(READ(ref/'selection.json')['selected']['diverse']);_durable_json(out/'selection.json',dict(selected=selected,reference_sha256=sha(ref/'selection.json')))
        def train_guard(device,sid):
            assert device=='EAST' and sid in selected and sid not in plan['held']
            access.append(dict(fold=f,shot=sid,phase='selected_preparation'))
            return reader(device,sid)
        with patch.object(io23,'read_outcome',side_effect=train_guard):
            target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in selected]
            training=BothBlockPreparation(copy.deepcopy(parent.base),'increment').cuda();training.load_state_dict(origin)
            prepare(training,source,target,out)
            for name in ('risk_manifest.jsonl','forecast_manifest.jsonl'):assert (out/name).read_bytes()==(ref/'diverse'/name).read_bytes()
            model=to_deployment(training);frozen=state_cpu(model);del training
        def deny(*args,**kwargs):raise AssertionError('labels forbidden during streaming prediction')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                scores,updates,_=predict(model,{k:s[k] for k in ('values','available')})
                np.testing.assert_array_equal(scores['static'][:256],scores['intra'][:256])
                curves={'both_'+k:v for k,v in scores.items()}
                with np.load(ref/'shots'/f'{sid}.npz') as old:
                    np.testing.assert_array_equal(old['times'],s['times'])
                    curves.update(source=old['source'].copy(),first_static=old['diverse_static'].copy(),first_intra=old['diverse_intra'].copy())
                path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
                np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
                receipts.append(dict(fold=f,shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
                steps+=sum(r['gradient_steps'] for r in updates)
                if len(receipts)%5==0:print(dict(stage='predict',shots=len(receipts),seconds=time.monotonic()-start),flush=True)
        assert_unchanged(model,frozen);assert_unchanged(parent,origin)
    _durable_json(root/'prediction_receipts.json',receipts);_durable_json(root/'training_label_access.json',access)
    results,folds=score(root,p['folds'],threshold)
    for a,b in (('source','source'),('first_static','diverse_static'),('first_intra','diverse_intra')):assert results[a]==reference['metrics'][b]
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics=results,fold_metrics=folds,seconds=time.monotonic()-start,
        offline_steps=4*(512+256),online_steps=steps,source_unchanged=True,
        independent_confirmation=False,test360_accessed=False,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
