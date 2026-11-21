"""Independent shot recount, acquisition checks, and selected full-shot replay."""
import argparse
import hashlib
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..increment_forecast_20261023.learning import predict
from ..simple_online import _durable_json
from ..simple_train import sha
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from .run import READ,ARMS,METHODS,make_parent,coverage
from .selection import describe,select,identity_folds


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for name,h in done['artifact_sha256'].items():assert sha(root/name)==h,name
    audit_name=str(Path(__file__).resolve().relative_to(io23.ROOT))
    # The executed experiment is checked against its immutable source archive.
    # Only this post-run verifier was optimized to read the static role list
    # once (196608 redundant JSON parses otherwise); science code stays exact.
    with zipfile.ZipFile(root/'code_snapshot.zip') as archive:
        for name,h in p['code_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==h,name
            if name!=audit_name:assert sha(io23.ROOT/name)==h,name
    for name,h in p['input_sha256'].items():assert sha(name)==h,name
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert identity_folds(p['shot_ids'])==p['folds']
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent();origin=state_cpu(parent)
    center=parent.base.center.detach().cpu().numpy();scale=parent.base.scale.detach().cpu().numpy()
    access=READ(root/'training_label_access.json');expected=READ(root/'per_shot.json')
    source_ids=set(io23.source_roles()['train'])
    counts={a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0) for a in ARMS};checks=0;replays=[];selected=[]
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';selection=READ(folder/'selection.json')
        with np.load(folder/'selection_features.npz') as z:
            ids=z['ids'].tolist();x=z['descriptors'].copy()
        assert ids==plan['candidate'] and not set(ids)&set(plan['held'])
        def deny(*args,**kwargs):raise AssertionError('labels forbidden during selection/replay')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for i,sid in enumerate(ids):
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                np.testing.assert_array_equal(describe({k:s[k] for k in ('values','available')},center,scale),x[i])
            for method in METHODS:
                chosen=select(ids,x,20,method,131+f)
                assert chosen==selection['selected'][method]
                assert coverage(x,ids,chosen)==selection['coverage'][method]
                used=[r['shot'] for r in access if r['fold']==f and r['method']==method]
                assert len(used)==20 and set(used)==set(chosen)
                prep=READ(folder/method/'preparation.json')
                assert prep['fit_ids']==sorted(chosen)
                selected.append(dict(fold=f,method=method,n_disrupt=prep['n_disrupt'],coverage=selection['coverage'][method]))
                # No unqueried historical shot may enter either training loss.
                for filename in ('risk_manifest.jsonl','forecast_manifest.jsonl'):
                    for line in (folder/method/filename).read_text().splitlines():
                        row=__import__('json').loads(line)
                        assert all(r['shot'] in chosen for r in row['target'])
                        assert all(r['shot'] in source_ids for r in row['source'])
        fold_counts={a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0) for a in ARMS};held=[]
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            held.append((sid,s['is_disrupt']))
            with np.load(folder/'shots'/f'{sid}.npz') as z:
                for a in ARMS:
                    prob=z[a];indices=np.flatnonzero(s['valid']&(prob>=p['threshold']));alarm=bool(len(indices))
                    lead=float(np.round((s['down']-s['times'][indices[0]])*1000,9)) if s['is_disrupt'] and alarm else None
                    timely=lead is not None and lead>30.
                    row=next(v for v in expected[a]['predictions'] if v['shot']==sid)
                    assert row==dict(shot=sid,is_disrupt=s['is_disrupt'],alarm=alarm,lead_ms=lead,timely=timely)
                    for c in (counts[a],fold_counts[a]):
                        c['n_disrupt']+=int(s['is_disrupt']);c['n_safe']+=int(not s['is_disrupt'])
                        c['tp']+=int(timely);c['fp']+=int(alarm and not s['is_disrupt'])
                    checks+=1
        for a in ARMS:
            for k,v in fold_counts[a].items():assert done['fold_metrics'][str(f)][a][k]==v
        if f==0:
            chosen_replay=[next((sid for sid,d in held if d==flag),None) for flag in (False,True)]
            for method in METHODS:
                model,_=make_parent();model.load_state_dict(torch.load(folder/method/'prepared.pt',map_location='cuda',weights_only=True)['model']);model.set_stage('frozen')
                frozen=state_cpu(model)
                assert_unchanged(model,origin,(*model.encoder_names(),'base.forecast_head.weight','base.forecast_head.bias',
                    'reconstructor.weight','reconstructor.bias','event_head.weight','event_head.bias'))
                with patch.object(io23,'read_outcome',side_effect=deny):
                    for sid in chosen_replay:
                        if sid is None:continue  # Only replay an actually present class; counts above disclose empty classes.
                        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                        curves,updates,_=predict(model,{k:s[k] for k in ('values','available')})
                        with np.load(folder/'shots'/f'{sid}.npz') as z:
                            for k,v in curves.items():np.testing.assert_array_equal(v,z[method+'_'+k])
                        assert updates==READ(folder/'shots'/f'{sid}_updates.json')[method]
                        replays.append(dict(fold=f,method=method,shot=sid))
                assert_unchanged(model,frozen)
        print(dict(fold=f,shot_arm_checks=checks,seconds=time.monotonic()-start),flush=True)
    for a in ARMS:
        for k,v in counts[a].items():assert expected[a][k]==done['metrics'][a][k]==v
        tpr=counts[a]['tp']/counts[a]['n_disrupt'];fpr=counts[a]['fp']/counts[a]['n_safe']
        assert expected[a]['tpr']==tpr and expected[a]['fpr']==fpr and expected[a]['J']==tpr-fpr
    assert_unchanged(parent,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_COUNTS_AND_SELECTED_REPLAY',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),shot_arm_checks=checks,replays=replays,selection=selected,
        archived_audit_sha256=p['code_sha256'][audit_name],
        verifier_revision='cache immutable source role membership once; training/inference source unchanged',
        seconds=time.monotonic()-start,source_unchanged=True,training_reproduced=False,
        independent_confirmation=False,test360_accessed=False)
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
