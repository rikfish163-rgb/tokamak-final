"""Independent shot-level recount and exact CUDA replay of the centered variant."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from ..internal_fast_window_20261023.audit import recount, correct
from ..internal_fast_window_20261023.learning import ARMS, predict
from .run import (io23,READ,sha,make_parent,models_for_calibration,REFERENCE,read_signal,
    transform,state_cpu,assert_unchanged,_durable_json,DEFAULT_PARENT,PARENT_SHA256,PREVIOUS)
from .model import CenteredAdapter


def main(root):
    p=READ(root/'protocol.json'); done=READ(root/'completed.json')
    for f,h in done['artifact_sha256'].items(): assert sha(root/f)==h,f
    for f,h in p['code_sha256'].items(): assert sha(io23.ROOT/f)==h,f
    for f,h in p['input_sha256'].items(): assert sha(f)==h,f
    stored=READ(root/'per_shot.json'); rows=READ(root/'outcomes.json'); seals=READ(root/'prediction_seals.json')
    assert p['shot_ids']==[r['shot'] for r in rows]==[r['shot'] for r in seals]
    counts={a:dict(tp=0,fp=0,very_early_over3000=0) for a in ARMS}; checks=0
    changes={a:dict(good=[],bad=[]) for a in ('parent','capacity','adaptive_off','adaptive_reverse')}
    inner=dict(endpoints=0,updated=0,loss_decreased=0,loss_increased=0)
    delta=[]; reverse=[]
    for index,(r,seal) in enumerate(zip(rows,seals)):
        sid=r['shot']; assert io23.read_outcome('EAST',sid)==(r['is_disrupt'],r['down'])
        path=root/'shots'/f'{sid}.npz'; trace=root/'shots'/f'{sid}_inner.npz'
        assert sha(path)==seal['prediction_sha256']==r['prediction_sha256']; assert sha(trace)==seal['inner_sha256']
        with np.load(path) as z,np.load(PREVIOUS/'shots'/f'{sid}.npz') as old:
            for a in ('parent','capacity'): np.testing.assert_array_equal(z[a],old[a])
            for a in ARMS:
                rr=dict(shot=sid,**recount(z['times'],z[a],r['is_disrupt'],r['down'],p['threshold']))
                assert rr==stored[a]['predictions'][index]; checks+=1
                counts[a]['tp']+=int(rr['timely']); counts[a]['fp']+=int(rr['alarm'] and not rr['is_disrupt'])
                counts[a]['very_early_over3000']+=int(rr['timely'] and rr['lead_ms']>3000)
            after=stored['adaptive_on']['predictions'][index]
            for a in changes:
                before=stored[a]['predictions'][index]
                if correct(before)!=correct(after): changes[a]['good' if correct(after) else 'bad'].append(dict(shot=sid,before=before,after=after))
            delta.append(float(np.max(np.abs(z['adaptive_on']-z['adaptive_off']))))
            reverse.append(float(np.max(np.abs(z['adaptive_on']-z['adaptive_reverse']))))
        with np.load(trace) as t:
            np.testing.assert_array_equal(t['updated'],t['observations']>=2)
            assert np.all(t['loss_after']<=t['loss_before']+1e-6+1e-5*t['loss_before'])
            for k in t.files: assert np.isfinite(t[k]).all()
            inner['endpoints']+=len(t['updated']); inner['updated']+=int(t['updated'].sum())
            inner['loss_decreased']+=int((t['loss_after']<t['loss_before']).sum())
            inner['loss_increased']+=int((t['loss_after']>t['loss_before']).sum())
    for a,cs in counts.items():
        for k,n in cs.items(): assert done['metrics'][a][k]==n
    torch.set_num_threads(2); torch.manual_seed(131); base,_=make_parent(); parent=models_for_calibration(base)['mapped_fixed']
    adaptive=CenteredAdapter(copy.deepcopy(parent.base)).cuda().eval(); capacity=copy.deepcopy(adaptive)
    layers=torch.load(root/'layers.pt',map_location='cpu',weights_only=True)
    for name,m in (('adaptive',adaptive),('capacity',capacity)): m.layer.load_state_dict(layers[name])
    states={name:state_cpu(m) for name,m in (('adaptive',adaptive),('capacity',capacity))}
    selected=list(dict.fromkeys([p['shot_ids'][0],p['shot_ids'][len(rows)//2],p['shot_ids'][-1]]+
        [r['shot'] for direction in ('good','bad') for r in changes['adaptive_off'][direction][:1]]))
    mapping=READ(REFERENCE/'input_map.json')
    def deny(*args,**kwargs): raise AssertionError('no outcome access during replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in selected:
            s=read_signal(sid); curves,traces=predict(adaptive,capacity,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for a in ARMS: np.testing.assert_array_equal(curves[a],z[a])
            with np.load(root/'shots'/f'{sid}_inner.npz') as z:
                for k,v in traces.items(): np.testing.assert_array_equal(v,z[k])
    for name,m in (('adaptive',adaptive),('capacity',capacity)): assert_unchanged(m,states[name])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    verification=dict(status='VERIFIED_CENTERED_WINDOW_EXPERIMENT',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),independent_recounts=checks,replayed_shots=selected,counts=counts,
        changes_vs_on=changes,inner=inner,median_per_shot_max_on_off_difference=float(np.median(delta)),
        median_per_shot_max_reverse_difference=float(np.median(reverse)),
        positive_intra_gain=done['metrics']['adaptive_on']['J']>done['metrics']['adaptive_off']['J'],
        improved_over_parent=done['metrics']['adaptive_on']['J']>done['metrics']['parent']['J'],
        fpr_cap_met=done['metrics']['adaptive_on']['fpr']<=.2,independent_confirmation=False)
    _durable_json(root/'verification.json',verification)
    print({k:v for k,v in verification.items() if k!='changes_vs_on'},flush=True)
    print({a:{k:len(v) for k,v in c.items()} for a,c in changes.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); main(parser.parse_args().root)
