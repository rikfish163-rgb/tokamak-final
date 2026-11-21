"""Recount all arms, refit the map, replay four shots and one full preparation."""
import argparse
import copy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.learning import SIGNAL_KEYS
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import make_parent,READ
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import sha
from ..temporal_scope_20261023.audit import recount
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many
from .mapping import fit_mapping,map_record
from .run import ARMS,METHODS,REFERENCE,REF_SHA,REUSED,prepare


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json');folder=root/'fold0'
    for name,h in done['artifact_sha256'].items():assert sha(root/name)==h,name
    for name,h in p['code_sha256'].items():assert sha(io23.ROOT/name)==h,name
    for name,h in {**p['input_sha256'],**p['shotlist_sha256']}.items():assert sha(name)==h,name
    assert sha(REFERENCE/'completed.json')==REF_SHA and sha(DEFAULT_PARENT)==PARENT_SHA256
    counts=recount(root,p,ARMS);assert counts['shot_arm_checks']==3600
    _durable_json(root/'threshold_diagnostic.json',counts)
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent();origin=state_cpu(parent)
    selected=p['selected'];assert selected==READ(REFERENCE/'protocol.json')['selected']
    assert len(selected)==20 and not set(selected)&set(p['shot_ids'])
    source_ids=io23.source_roles()['train'];reader=io23.read_outcome
    def train_guard(device,sid):
        assert (device=='J-TEXT' and sid in source_ids) or (device=='EAST' and sid in selected)
        return reader(device,sid)
    def deny(*args,**kwargs):raise AssertionError('outcomes forbidden during map fitting and replay')
    with patch.object(io23,'read_outcome',side_effect=train_guard):source,_=load_source_train()
    with patch.object(io23,'read_outcome',side_effect=deny):
        raw_target=[apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')) for sid in selected]
        mapping=fit_mapping([{k:s[k] for k in SIGNAL_KEYS} for s in source],
            [{k:s[k] for k in SIGNAL_KEYS} for s in raw_target],source_ids=source_ids,target_ids=selected)
    assert mapping==READ(root/'input_map.json')
    access=READ(root/'training_label_access.json');replays=[]
    for method in METHODS:
        out=folder/method;assert [r['shot'] for r in access if r['method']==method]==selected
        for name in ('risk_manifest.jsonl','forecast_manifest.jsonl'):
            assert (out/name).read_bytes()==(REFERENCE/'fold0/first_head'/name).read_bytes()
        model=copy.deepcopy(parent);model.load_state_dict(torch.load(out/'prepared.pt',map_location='cuda',weights_only=True)['model']);model.set_stage('frozen')
        allowed=(*model.encoder_names(),'base.forecast_head.weight','base.forecast_head.bias')
        if method=='mapped_head':allowed+=('base.risk_head.weight','base.risk_head.bias')
        assert_unchanged(model,origin,allowed);before=state_cpu(model)
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in (54157,54160):
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                scores,updates,_=predict(model,signal)
                with np.load(folder/'shots'/f'{sid}.npz') as z:
                    for a,v in scores.items():np.testing.assert_array_equal(v,z[method+'_'+a])
                    with np.load(REFERENCE/'fold0/shots'/f'{sid}.npz') as old:
                        for a,b in REUSED.items():np.testing.assert_array_equal(z[a],old[b])
                    if method==METHODS[0]:np.testing.assert_array_equal(z['mapped_source_static'],predict_many(parent,dict(static=signal),device='cuda')['static'])
                assert updates==READ(folder/'shots'/f'{sid}_updates.json')[method]
                replays.append(dict(method=method,shot=sid))
        assert_unchanged(model,before)
    repeat=root/'refit_mapped_head';repeat.mkdir()
    with patch.object(io23,'read_outcome',side_effect=train_guard):
        target=[map_record(simple_data.attach_outcome(s,'EAST',{}),mapping) for s in raw_target]
        model=copy.deepcopy(parent);prepare(model,source,target,repeat,'mapped_head')
    saved=torch.load(folder/'mapped_head/prepared.pt',map_location='cpu',weights_only=True)['model']
    for k,v in model.state_dict().items():assert torch.equal(v.cpu(),saved[k]),k
    for name in ('risk_manifest.jsonl','forecast_manifest.jsonl','risk_training.json','forecast_training.json','preparation.json'):
        assert (repeat/name).read_bytes()==(folder/'mapped_head'/name).read_bytes(),name
    assert_unchanged(parent,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_COUNTS_MAP_REPLAY_AND_MAPPED_HEAD_REFIT',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),diagnostic_sha256=sha(root/'threshold_diagnostic.json'),shot_arm_checks=3600,
        map_exact_refit=True,training_manifests_exact_match=4,replays=replays,complete_preparation_refit='mapped_head',
        refit_steps=768,source_unchanged=True,seconds=time.monotonic()-start,
        refit_sha256={str(v.relative_to(root)):sha(v) for v in repeat.iterdir() if v.is_file()})
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
