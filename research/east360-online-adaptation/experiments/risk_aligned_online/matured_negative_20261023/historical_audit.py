"""Recount the historical control and reproduce two uncached full-shot runs."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..heldout_threshold_20261023.run import models_for_calibration
from ..history_tangent_20261023.learning import logit
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .historical_control import PRIMARY,REFERENCE,negative_replay,historical_step
from .learning import tensors
from .run import historical_replay


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    p=READ(root/'protocol.json');done=READ(root/'completed.json');rows=READ(root/'per_shot.json')
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    assert sha(PRIMARY/'completed.json')==p['primary_sha256']
    counts={level:dict(tp=0,fp=0,very_early_over3000=0) for level in rows}
    byid={level:{r['shot']:r for r in value['predictions']} for level,value in rows.items()}
    checks=0
    for sid in p['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz') as z:
            for level,row in rows.items():
                ids=np.flatnonzero(s['valid'] & (z['scores']>=row['threshold']))
                alarm=bool(len(ids));lead=float(round((s['down']-s['times'][ids[0]])*1000,9)) if alarm and s['is_disrupt'] else None
                r=dict(shot=sid,is_disrupt=s['is_disrupt'],alarm=alarm,lead_ms=lead,timely=lead is not None and lead>30)
                assert r==byid[level][sid],(level,sid)
                counts[level]['tp']+=int(r['timely']);counts[level]['fp']+=int(alarm and not s['is_disrupt'])
                counts[level]['very_early_over3000']+=int(r['timely'] and lead>3000);checks+=1
    for level,c in counts.items():
        assert all(rows[level][k]==v for k,v in c.items())
        assert rows[level]['tpr']==c['tp']/180 and rows[level]['fpr']==c['fp']/180
    primary=READ(PRIMARY/'per_shot.json');flips={}
    def correct(r):return r['timely'] if r['is_disrupt'] else not r['alarm']
    for level in rows:
        for arm in ('static','matured_plus_replay'):
            changes=[]
            for b in primary[arm+'/'+level]['predictions']:
                a=byid[level][b['shot']]
                if correct(a)!=correct(b):changes.append(dict(shot=a['shot'],beneficial=correct(b),historical=a,other=b))
            flips['historical -> '+arm+'/'+level]=changes
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent()
    model=models_for_calibration(parent)['mapped_fixed'];initial=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json')
    positive,pm=historical_replay(READ(PRIMARY/'protocol.json')['preparation_ids'],mapping)
    negative,nm=negative_replay(pm,mapping)
    assert pm==p['positive_manifest'] and nm==p['negative_manifest']
    receipts=READ(root/'cached_update_sequence.json');updates_checked=0
    def deny(*args,**kwargs):raise AssertionError('outcome forbidden during replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in (63690,64581):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=dict(transform({k:s[k] for k in ('values','available')},mapping),times=s['times'])
            fast=initial_encoder(model);scores=[];i=0
            for lo in range(0,len(s['times']),128):
                hi=min(lo+128,len(s['times']))
                for start in range(lo,hi,32):
                    x,m=tensors(signal,range(start,min(start+32,hi)),'cuda')
                    with torch.no_grad():scores.append(logit(model,x,m,fast).sigmoid().cpu().numpy())
                if hi>=256 and hi<len(s['times']):
                    fast,r=historical_step(model,fast,negative,positive)
                    assert r==receipts[i];i+=1;updates_checked+=1
            with np.load(root/'shots'/f'{sid}.npz') as z:np.testing.assert_array_equal(np.concatenate(scores),z['scores'])
    assert_unchanged(model,initial);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_RECOUNT_AND_UNCACHED_REPLAYS',completed_sha256=sha(root/'completed.json'),
                audit_sha256=sha(Path(__file__)),shot_arm_checks=checks,counts=counts,flips=flips,
                replayed_shots=[63690,64581],replayed_updates=updates_checked,source_unchanged=True)
    _durable_json(root/'verification.json',result)
    print({k:v for k,v in result.items() if k!='flips'},flush=True)
    print({k:dict(beneficial=sum(r['beneficial'] for r in v),harmful=sum(not r['beneficial'] for r in v)) for k,v in flips.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
