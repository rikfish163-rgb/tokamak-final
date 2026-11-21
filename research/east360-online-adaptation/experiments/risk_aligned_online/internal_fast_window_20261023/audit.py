"""Independent first-alarm recount and real CUDA replay of all five arms."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..contextual_feedback_20261023.run import read_signal
from ..continual_episode_20261023.audit import recount
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..heldout_threshold_20261023.run import REFERENCE,models_for_calibration
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS,predict
from .model import WindowAdapter


def correct(r):return r['timely'] if r['is_disrupt'] else not r['alarm']


def main(root):
    p=READ(root/'protocol.json');done=READ(root/'completed.json');stored=READ(root/'per_shot.json')
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    for f,h in p['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    for f,h in p['input_sha256'].items():assert sha(f)==h,f
    outcomes=READ(root/'outcomes.json');seals=READ(root/'prediction_seals.json')
    assert [r['shot'] for r in outcomes]==p['shot_ids']==[r['shot'] for r in seals]
    assert not set(p['preparation_ids'])&set(p['shot_ids'])
    assert not set(p['calibration_ids'])&(set(p['shot_ids'])|set(p['preparation_ids']))
    train=READ(root/'training.json')
    assert len(train)==p['additional_preparation_steps']
    for r in train:
        assert len(r['samples'])==32 and sum(x['label'] for x in r['samples'])==16
        assert {x['shot'] for x in r['samples']}<=set(p['preparation_ids'])
    counts={a:dict(tp=0,fp=0,very_early_over3000=0) for a in ARMS};checks=0
    changes={a:dict(good=[],bad=[]) for a in ('parent','capacity','adaptive_off','adaptive_reverse')}
    inner=dict(endpoints=0,updated=0,loss_decreased=0,loss_increased=0,max_delta_norm=0.,max_residual_norm=0.)
    pair_magnitudes={a:[] for a in changes}
    for index,(o,seal) in enumerate(zip(outcomes,seals)):
        sid=o['shot'];assert io23.read_outcome('EAST',sid)==(o['is_disrupt'],o['down'])
        assert sha(root/'shots'/f'{sid}.npz')==seal['prediction_sha256']==o['prediction_sha256']
        assert sha(root/'shots'/f'{sid}_inner.npz')==seal['inner_sha256']
        with np.load(root/'shots'/f'{sid}.npz') as z:
            for a in ARMS:
                r=dict(shot=sid,**recount(z['times'],z[a],o['is_disrupt'],o['down'],p['threshold']))
                assert r==stored[a]['predictions'][index],(sid,a)
                counts[a]['tp']+=int(r['timely']);counts[a]['fp']+=int(r['alarm'] and not r['is_disrupt'])
                counts[a]['very_early_over3000']+=int(r['timely'] and r['lead_ms']>3000);checks+=1
            current=stored['adaptive_on']['predictions'][index]
            for a in changes:
                before=stored[a]['predictions'][index]
                if correct(before)!=correct(current):
                    changes[a]['good' if correct(current) else 'bad'].append(dict(shot=sid,before=before,after=current))
                pair_magnitudes[a].append(float(np.max(np.abs(z['adaptive_on']-z[a]))))
        with np.load(root/'shots'/f'{sid}_inner.npz') as t:
            assert len(t['updated'])==len(t['observations'])
            np.testing.assert_array_equal(t['updated'],t['observations']>=2)
            for name in t.files:assert np.isfinite(t[name]).all()
            inner['endpoints']+=len(t['updated']);inner['updated']+=int(t['updated'].sum())
            inner['loss_decreased']+=int((t['loss_after']<t['loss_before']).sum())
            inner['loss_increased']+=int((t['loss_after']>t['loss_before']).sum())
            inner['max_delta_norm']=max(inner['max_delta_norm'],float(t['delta_norm'].max()))
            inner['max_residual_norm']=max(inner['max_residual_norm'],float(t['residual_norm'].max()))
    for a,c in counts.items():
        for k,v in c.items():assert v==done['metrics'][a][k]
    torch.set_num_threads(2);torch.manual_seed(131)
    base,_=make_parent();parent=models_for_calibration(base)['mapped_fixed']
    adaptive=WindowAdapter(copy.deepcopy(parent.base)).cuda().eval();capacity=copy.deepcopy(adaptive)
    layer=torch.load(root/'layers.pt',map_location='cpu',weights_only=True)
    for name,net in (('capacity',capacity),('adaptive',adaptive)):net.layer.load_state_dict(layer[name])
    snapshot=state_cpu(adaptive);cap_snapshot=state_cpu(capacity)
    mapping=READ(REFERENCE/'input_map.json');replayed=[]
    def deny(*args,**kwargs):raise AssertionError('no outcome in replayed prediction')
    selected=list(dict.fromkeys([p['shot_ids'][0],p['shot_ids'][len(p['shot_ids'])//2],p['shot_ids'][-1]]+
        [r['shot'] for direction in ('good','bad') for r in changes['adaptive_off'][direction][:1]]))
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in selected:
            s=read_signal(sid);signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,traces=predict(adaptive,capacity,signal)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for a in ARMS:np.testing.assert_array_equal(curves[a],z[a])
            with np.load(root/'shots'/f'{sid}_inner.npz') as z:
                for k,v in traces.items():np.testing.assert_array_equal(v,z[k])
            replayed.append(sid)
    assert_unchanged(adaptive,snapshot);assert_unchanged(capacity,cap_snapshot)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_INTERNAL_WINDOW_EXPERIMENT',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),independent_recounts=checks,replayed_shots=replayed,counts=counts,
        changes_vs_on=changes,inner=inner,median_per_shot_max_probability_change={k:float(np.median(v)) for k,v in pair_magnitudes.items()},
        initial_to_final_parameter_norm={name:float(sum((v-layer['initial'][k]).double().square().sum() for k,v in layer[name].items()).sqrt()) for name in ('capacity','adaptive')},
        positive_intra_gain=done['metrics']['adaptive_on']['J']>done['metrics']['adaptive_off']['J'],
        fpr_cap_met=done['metrics']['adaptive_on']['fpr']<=.2,
        independent_confirmation=False)
    _durable_json(root/'verification.json',result)
    print({k:v for k,v in result.items() if k!='changes_vs_on'},flush=True)
    print({a:{k:len(v) for k,v in r.items()} for a,r in changes.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
