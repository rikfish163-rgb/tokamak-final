"""Recompute selection/counts, verify matched batches, and replay/refit fold0."""
import argparse
import copy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import make_parent,READ
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import sha
from ..temporal_scope_20261023.audit import recount
from .design import class_aware_selection
from .learning import prepare
from .run import ARMS,NEW,REFERENCE,REF_SHA,SHOTLIST


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for name,h in done['artifact_sha256'].items():assert sha(root/name)==h,name
    for name,h in p['code_sha256'].items():assert sha(io23.ROOT/name)==h,name
    for name,h in {**p['input_sha256'],**p['shotlist_sha256']}.items():assert sha(name)==h,name
    assert sha(REFERENCE/'completed.json')==REF_SHA and sha(DEFAULT_PARENT)==PARENT_SHA256
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent();origin=state_cpu(parent)
    counts=recount(root,p,ARMS);_durable_json(root/'threshold_diagnostic.json',counts)
    source_ids=set(io23.source_roles()['train']);disruptive,safe=[set(READ(v)) for v in SHOTLIST]
    access=READ(root/'training_label_access.json');replays=[];classes=[];manifests=0
    for plan in p['folds']:
        f=plan['fold'];out=root/f'fold{f}';selection=READ(out/'selection.json')
        with np.load(REFERENCE/f'fold{f}'/'selection_features.npz') as z:
            ids=z['ids'].tolist();descriptors=z['descriptors'].copy()
        assert ids==plan['candidate']
        labels={sid:sid in disruptive for sid in ids}
        assert all((sid in disruptive)!=(sid in safe) for sid in ids)
        assert {int(k):v for k,v in selection['candidate_class_access'].items()}==labels
        assert selection['selected']['aware']==class_aware_selection(ids,descriptors,labels)
        assert selection['selected']['blind']==sorted(READ(REFERENCE/f'fold{f}'/'selection.json')['selected']['diverse'])
        for method in NEW:
            folder=out/method;selected=selection['selected']['blind' if method.startswith('blind') else 'aware']
            assert len(selected)==20 and not set(selected)&set(plan['held'])
            assert [r['shot'] for r in access if r['fold']==f and r['method']==method]==selected
            target={sid:simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in selected}
            assert all(s['is_disrupt']==labels[sid] for sid,s in target.items())
            record=READ(folder/'preparation.json');assert record['fit_ids']==selected
            assert record['use_source']==method.endswith('mixed')
            assert record['n_disrupt']==sum(s['is_disrupt'] for s in target.values())
            if method.startswith('aware'):assert record['n_disrupt']==4
            classes.append(dict(fold=f,method=method,n_disrupt=record['n_disrupt']))
            for filename in ('risk_manifest.jsonl','forecast_manifest.jsonl'):
                records=[__import__('json').loads(line) for line in (folder/filename).read_text().splitlines()]
                assert len(records)==(512 if filename.startswith('risk') else 256)
                for row in records:
                    assert all(r['shot'] in source_ids for r in row['source'])
                    for r in row['target']:
                        assert r['shot'] in target
                        if filename.startswith('risk'):
                            s=target[r['shot']];assert s['valid'][r['end']]
                            assert r['label']==int(s['y'][r['end']]>.5)
                manifests+=1
            other=REFERENCE/f'fold{f}'/'diverse' if method=='blind_target' else out/'aware_mixed'
            if method!='aware_mixed':
                for filename in ('risk_manifest.jsonl','forecast_manifest.jsonl'):assert (folder/filename).read_bytes()==(other/filename).read_bytes()
            relations=READ(folder/'gradient_relations.json')
            assert [r['step'] for r in relations]==list(range(63,512,64))
            model=copy.deepcopy(parent)
            model.load_state_dict(torch.load(folder/'prepared.pt',map_location='cuda',weights_only=True)['model']);model.set_stage('frozen')
            assert_unchanged(model,origin,(*model.encoder_names(),'base.forecast_head.weight','base.forecast_head.bias'))
            if f==0:
                before=state_cpu(model)
                def deny(*args,**kwargs):raise AssertionError('no outcome access during replay')
                with patch.object(io23,'read_outcome',side_effect=deny):
                    for sid in (54314,63226):
                        assert sid in plan['held']
                        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                        scores,updates,_=predict(model,{k:s[k] for k in ('values','available')})
                        with np.load(out/'shots'/f'{sid}.npz') as z:
                            for a,v in scores.items():np.testing.assert_array_equal(v,z[method+'_'+a])
                        assert updates==READ(out/'shots'/f'{sid}_updates.json')[method]
                        replays.append(dict(fold=f,method=method,shot=sid))
                assert_unchanged(model,before)
        print(dict(stage='audit_fold',fold=f,seconds=time.monotonic()-start),flush=True)
    # Full preparation replay and control-equivalence check. These are verifier
    # runs, not additional models selected using held-out performance.
    reader=io23.read_outcome
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    selected=READ(root/'fold0'/'selection.json')['selected']['blind']
    def train_guard(device,sid):
        assert device=='EAST' and sid in selected
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=train_guard):
        target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in selected]
        refits=[]
        for use_source,name,ref in ((False,'blind_target',root/'fold0'/'blind_target'),(True,'blind_mixed',REFERENCE/'fold0'/'diverse')):
            folder=root/('audit_refit0_'+name);folder.mkdir();model=copy.deepcopy(parent)
            prepare(model,source,target,folder,use_source=use_source)
            saved=torch.load(ref/'prepared.pt',map_location='cpu',weights_only=True)['model']
            for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),saved[k],rtol=0,atol=0)
            for filename in ('risk_manifest.jsonl','forecast_manifest.jsonl','risk_training.json','forecast_training.json'):
                assert (folder/filename).read_bytes()==(ref/filename).read_bytes()
            if not use_source:assert (folder/'gradient_relations.json').read_bytes()==(ref/'gradient_relations.json').read_bytes()
            refits.append(dict(method=name,state_exact=True,batches_and_losses_exact=True,
                artifacts={str(v.relative_to(root)):sha(v) for v in folder.iterdir() if v.is_file()}))
    assert_unchanged(parent,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_COUNTS_SELECTION_SELECTED_REPLAY_AND_REFIT',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),diagnostic_sha256=sha(root/'threshold_diagnostic.json'),
        shot_arm_checks=counts['shot_arm_checks'],training_manifests=manifests,classes=classes,
        replays=replays,refits=refits,source_unchanged=True,seconds=time.monotonic()-start)
    _durable_json(root/'verification.json',result);print({k:v for k,v in result.items() if k!='refits'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
