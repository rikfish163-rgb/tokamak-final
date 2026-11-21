"""Independent event counts, matched doses and four complete CUDA replays."""
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
from ..matured_negative_20261023.historical_control import PRIMARY,REFERENCE,negative_replay
from ..matured_negative_20261023.run import historical_replay
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS,predict


def correct(r):return r['timely'] if r['is_disrupt'] else not r['alarm']


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    p=READ(root/'protocol.json');done=READ(root/'completed.json');rows=READ(root/'per_shot.json')
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['input_sha256'].items():assert sha(name)==digest,name
    assert sha(PRIMARY/'completed.json')==p['primary_sha256']
    byid={k:{r['shot']:r for r in v['predictions']} for k,v in rows.items()}
    counts={k:dict(tp=0,fp=0,very_early_over3000=0) for k in rows}
    horizons={a:dict(has_30_125=0,early_only=0) for a in p['arms']}
    stats={a:dict(steps=0,nonzero=0,objective_decreased=0,mean_positive_bce_increased=0,any_positive_bce_increased=0) for a in ARMS}
    checks=0;mature_uses=0;max_step_difference=0.;positive_changes={a:[] for a in ARMS}
    for sid in p['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz') as z:
            assert not s['is_disrupt'] or (z['times']<s['down']).all()
            leads=np.round((s['down']-z['times'])*1000,9) if s['is_disrupt'] else None
            for key in rows:
                arm,level=key.split('/');mask=s['valid']&(z[arm]>=rows[key]['threshold']);ids=np.flatnonzero(mask)
                alarm=bool(len(ids));lead=float(round((s['down']-z['times'][ids[0]])*1000,9)) if alarm and s['is_disrupt'] else None
                r=dict(shot=sid,is_disrupt=s['is_disrupt'],alarm=alarm,lead_ms=lead,timely=lead is not None and lead>30)
                assert r==byid[key][sid],(sid,key)
                counts[key]['tp']+=int(r['timely']);counts[key]['fp']+=int(alarm and not s['is_disrupt'])
                counts[key]['very_early_over3000']+=int(r['timely'] and lead>3000);checks+=1
                if s['is_disrupt'] and level=='calibrated':
                    near=(leads>30)&(leads<=125)
                    horizons[arm]['has_30_125']+=int((mask&near).any())
                    horizons[arm]['early_only']+=int((mask&(leads>125)).any() and not (mask&near).any())
        updates=READ(root/'shots'/f'{sid}_updates.json')
        assert len({len(v) for v in updates.values()})==1
        for i in range(len(updates[ARMS[0]])):
            reference=updates['rank_current'][i]['parameter_delta_norm']
            for arm in ARMS:
                r=updates[arm][i];count=r['first_affected_prediction'];now=float(s['times'][count-1])
                assert r['certified_event_free_through']==now
                assert r['last_observation']==count-1
                for e in r['negative_indices']:
                    age=round((now-float(s['times'][e]))*1000,9)
                    assert e>=127 and e<count and 125<=age<253 and s['valid'][e] and s['y'][e]==0
                    mature_uses+=1
                assert abs(r['parameter_delta_norm']-reference)<=2e-7
                max_step_difference=max(max_step_difference,abs(r['parameter_delta_norm']-reference))
                stats[arm]['steps']+=r['gradient_steps'];stats[arm]['nonzero']+=int(r['parameter_delta_norm']>0)
                if r['gradient_steps']:
                    stats[arm]['objective_decreased']+=int(r['loss_after']<r['loss_before'])
                    change=np.asarray(r['positive_bce_after'])-np.asarray(r['positive_bce_before'])
                    positive_changes[arm].append(float(change.mean()))
                    stats[arm]['mean_positive_bce_increased']+=int(change.mean()>1e-6)
                    stats[arm]['any_positive_bce_increased']+=int((change>1e-6).any())
    for key,c in counts.items():
        assert all(rows[key][k]==v for k,v in c.items()),key
        assert rows[key]['tpr']==c['tp']/180 and rows[key]['fpr']==c['fp']/180
    assert horizons==READ(root/'horizon_diagnosis.json')['horizon_counts']
    for arm in ARMS:
        stats[arm]['mean_positive_bce_change']=float(np.mean(positive_changes[arm]))
        assert stats[arm]['steps']==done['steps'][arm]
    flips={}
    for level in ('calibrated','source_threshold'):
        for before in ('static','previous_bce','bce_matched','rank_history_matched'):
            key=before+' -> rank_current/'+level;changes=[]
            for sid in p['shot_ids']:
                a=byid[before+'/'+level][sid];b=byid['rank_current/'+level][sid]
                if correct(a)!=correct(b):changes.append(dict(shot=sid,beneficial=correct(b),before=a,after=b))
            flips[key]=changes
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent()
    model=models_for_calibration(parent)['mapped_fixed'];initial=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');positive,pm=historical_replay(p['preparation_ids'],mapping)
    negative,nm=negative_replay(pm,mapping);assert pm==p['positive_manifest'] and nm==p['negative_manifest']
    def deny(*args,**kwargs):raise AssertionError('outcome forbidden during replay')
    replayed=[62765,64581,71415,87537]
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in replayed:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=dict(transform({k:s[k] for k in ('values','available')},mapping),times=s['times'])
            curves,rs=predict(model,signal,negative,positive)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for arm,curve in curves.items():np.testing.assert_array_equal(curve,z[arm])
            assert rs==READ(root/'shots'/f'{sid}_updates.json')
    assert_unchanged(model,initial);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_COUNTS_MATURITY_DOSES_AND_FOUR_REPLAYS',completed_sha256=sha(root/'completed.json'),
                audit_sha256=sha(Path(__file__)),shot_arm_checks=checks,mature_negative_uses_checked=mature_uses,
                max_step_difference=max_step_difference,counts=counts,horizon_counts=horizons,updates=stats,
                flips=flips,replayed_shots=replayed,source_unchanged=True,independent_confirmation=False)
    _durable_json(root/'verification.json',result)
    print({k:v for k,v in result.items() if k!='flips'},flush=True)
    print({k:dict(beneficial=sum(r['beneficial'] for r in v),harmful=sum(not r['beneficial'] for r in v)) for k,v in flips.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
