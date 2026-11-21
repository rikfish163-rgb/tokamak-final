"""Independent counts and selected complete replays for fixed20/TEST360."""
import argparse
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import make_parent,READ
from ..diverse_support_20261023.selection import describe
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..temporal_scope_20261023.audit import recount
from ..warmstart_alignment_20261022.run import predict_many
from .run import METHODS,ARMS,SHOTLIST
from .test_stream import select_preparation


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();done=READ(root/'completed.json');p=READ(root/'protocol.json');folder=root/'fold0'
    for name,h in done['artifact_sha256'].items():assert sha(root/name)==h,name
    for name,h in p['code_sha256'].items():assert sha(io23.ROOT/name)==h,name
    for name,h in {**p['input_sha256'],**p['shotlist_sha256']}.items():assert sha(name)==h,name
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    counts=recount(root,p,ARMS);assert counts['shot_arm_checks']==3240
    _durable_json(root/'threshold_diagnostic.json',counts)
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent();origin=state_cpu(parent)
    center=parent.base.center.detach().cpu().numpy();scale=parent.base.scale.detach().cpu().numpy()
    ids=p['history_ids'];held=p['shot_ids'];disruptive,safe=[set(READ(v)) for v in SHOTLIST]
    labels={s:s in disruptive for s in ids};assert all((s in disruptive)!=(s in safe) for s in ids)
    selection=READ(folder/'selection.json');assert {int(k):v for k,v in selection['candidate_class_access'].items()}==labels
    def deny(*args,**kwargs):raise AssertionError('no outcomes for descriptor or prediction replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        x=[]
        for sid in ids:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            x.append(describe({k:s[k] for k in ('values','available')},center,scale))
    x=np.stack(x)
    with np.load(folder/'selection_features.npz') as z:
        assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['descriptors'],x)
    selected=select_preparation(ids,x,labels,held);assert selected==selection['selected']
    access=READ(root/'training_label_access.json');source_ids=set(io23.source_roles()['train']);replays=[];manifests=0
    for method in METHODS:
        out=folder/method;fit=selected[method.split('_')[0]]
        assert [r['shot'] for r in access if r['method']==method]==fit
        target={sid:simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in fit}
        assert all(s['is_disrupt']==labels[sid] for sid,s in target.items())
        for name,n in (('risk_manifest.jsonl',512),('forecast_manifest.jsonl',256)):
            records=[json.loads(v) for v in (out/name).read_text().splitlines()];assert len(records)==n
            for row in records:
                assert all(v['shot'] in source_ids for v in row['source'])
                for v in row['target']:
                    assert v['shot'] in target
                    if name.startswith('risk'):
                        s=target[v['shot']];assert s['valid'][v['end']]
                        assert v['label']==int(s['y'][v['end']]>.5)
            manifests+=1
            if method.endswith('target'):
                other=folder/(method.split('_')[0]+'_mixed')
                assert (out/name).read_bytes()==(other/name).read_bytes()
        model=copy.deepcopy(parent);model.load_state_dict(torch.load(out/'prepared.pt',map_location='cuda',weights_only=True)['model']);model.set_stage('frozen')
        assert_unchanged(model,origin,(*model.encoder_names(),'base.forecast_head.weight','base.forecast_head.bias'))
        before=state_cpu(model)
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in held[:2]:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'));signal={k:s[k] for k in ('values','available')}
                scores,updates,_=predict(model,signal)
                with np.load(folder/'shots'/f'{sid}.npz') as z:
                    for a,v in scores.items():np.testing.assert_array_equal(v,z[method+'_'+a])
                    if method==METHODS[0]:np.testing.assert_array_equal(z['source'],predict_many(parent,dict(source=signal),device='cuda')['source'])
                assert updates==READ(folder/'shots'/f'{sid}_updates.json')[method]
                replays.append(dict(method=method,shot=sid))
        assert_unchanged(model,before)
    assert_unchanged(parent,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_COUNTS_SELECTION_AND_SELECTED_REPLAY',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),diagnostic_sha256=sha(root/'threshold_diagnostic.json'),
        shot_arm_checks=counts['shot_arm_checks'],training_manifests=manifests,replays=replays,
        preparation_refit=False,source_unchanged=True,seconds=time.monotonic()-start)
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
