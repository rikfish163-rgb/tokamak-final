"""Recount the sham ablation and replay the two prespecified previous flips."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import READ, make_parent
from ..heldout_threshold_20261023.run import models_for_calibration
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import REFERENCE,CALIBRATION
from .control import predict_sham


def main(root):
    if (root/'verification.json').exists(): raise FileExistsError(root/'verification.json')
    p=READ(root/'protocol.json');done=READ(root/'completed.json');rows=READ(root/'per_shot.json')
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['input_sha256'].items():assert sha(name)==digest,name
    assert p['threshold']==READ(CALIBRATION/'thresholds.json')['mapped_fixed']['0.2']['threshold']
    counts={k:dict(tp=0,fp=0) for k in rows};checks=0
    for sid in p['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz') as z:
            for key in rows:
                a,level=key.split('/');idx=np.flatnonzero(s['valid']&(z[a]>=rows[key]['threshold']))
                alarm=bool(len(idx));lead=float(round((s['down']-s['times'][idx[0]])*1000,9)) if s['is_disrupt'] and alarm else None
                r=dict(shot=sid,is_disrupt=s['is_disrupt'],alarm=alarm,lead_ms=lead,timely=lead is not None and lead>30.)
                assert r==next(v for v in rows[key]['predictions'] if v['shot']==sid)
                counts[key]['tp']+=int(r['timely']);counts[key]['fp']+=int(alarm and not s['is_disrupt']);checks+=1
    for k,c in counts.items():assert all(rows[k][name]==v for name,v in c.items())
    flips={}
    for level in ('calibrated','source_threshold'):
        for before,after in (('static','real_intra'),('static','persistence_sham'),('real_intra','persistence_sham')):
            old={r['shot']:r for r in rows[before+'/'+level]['predictions']};new={r['shot']:r for r in rows[after+'/'+level]['predictions']};changed=[]
            for sid,o in old.items():
                n=new[sid];good=lambda v:v['timely'] if v['is_disrupt'] else not v['alarm']
                if good(o)!=good(n):changed.append(dict(shot=sid,beneficial=good(n),before=o,after=n))
            flips[before+' -> '+after+'/'+level]=changed
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed']
    initial=state_cpu(model);mapping=READ(REFERENCE/'input_map.json');replays=[]
    def deny(*args,**kwargs):raise AssertionError('outcomes forbidden during control replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in (62765,87537):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            curves,updates,_=predict_sham(model,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(curves['static'],z['static'])
                np.testing.assert_array_equal(curves['intra'],z['persistence_sham'])
            assert updates==READ(root/'shots'/f'{sid}_updates.json');replays.append(sid)
    assert_unchanged(model,initial);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    updates=[u for sid in p['shot_ids'] for u in READ(root/'shots'/f'{sid}_updates.json')]
    result=dict(status='VERIFIED_COUNTS_AND_TWO_COMPLETE_SHAM_REPLAYS',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),shot_arm_checks=checks,flips=flips,replayed_shots=replays,
        updates=dict(count=len(updates),steps=sum(u['gradient_steps'] for u in updates),
            nonzero=sum(u['parameter_delta_norm']>0 for u in updates),loss_decreased=sum(u.get('loss_after',0)<u.get('loss_before',0) for u in updates)),
        source_unchanged=True,note='Post-run audit code separately hashed; two shots chosen before sham execution based on previous experiment.')
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
