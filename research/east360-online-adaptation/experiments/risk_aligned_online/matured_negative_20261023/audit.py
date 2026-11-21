"""Independent first-alarm recount, maturity checks and two full CUDA replays."""
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
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import MODES,predict_conditional
from .run import REFERENCE,CALIBRATION,historical_replay


def correct(row):
    return row['timely'] if row['is_disrupt'] else not row['alarm']


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    p=READ(root/'protocol.json');done=READ(root/'completed.json');rows=READ(root/'per_shot.json')
    assert p['loss_coefficients']==dict(negative=.5,replay=.5)
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['input_sha256'].items():assert sha(name)==digest,name
    assert p['threshold']==READ(CALIBRATION/'thresholds.json')['mapped_fixed']['0.2']['threshold']
    counts={k:dict(tp=0,fp=0,very_early_over3000=0) for k in rows}
    byid={k:{v['shot']:v for v in r['predictions']} for k,r in rows.items()}
    checks=0;mature_uses=0;updates={a:dict(steps=0,nonzero=0,loss_decreased=0,clipped=0) for a in MODES}
    early_fp={level:[] for level in ('calibrated','source_threshold')}
    for sid in p['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        assert not s['is_disrupt'] or (s['times']<s['down']).all()
        with np.load(root/'shots'/f'{sid}.npz') as z:
            for key in rows:
                arm,level=key.split('/');ids=np.flatnonzero(s['valid'] & (z[arm]>=rows[key]['threshold']))
                alarm=bool(len(ids));lead=float(round((s['down']-s['times'][ids[0]])*1000,9)) if s['is_disrupt'] and alarm else None
                r=dict(shot=sid,is_disrupt=s['is_disrupt'],alarm=alarm,lead_ms=lead,timely=lead is not None and lead>30.)
                assert r==byid[key][sid],(key,sid)
                counts[key]['tp']+=int(r['timely']);counts[key]['fp']+=int(alarm and not s['is_disrupt'])
                counts[key]['very_early_over3000']+=int(r['timely'] and lead>3000)
                if arm=='static' and alarm and not s['is_disrupt'] and ids[0]<256:
                    early_fp[level].append(dict(shot=sid,index=int(ids[0]),time=float(s['times'][ids[0]])))
                checks+=1
        receipts=READ(root/'shots'/f'{sid}_updates.json')
        for arm in MODES:
            for r in receipts[arm]:
                count=r['first_affected_prediction'];now=float(s['times'][count-1])
                assert r['certified_event_free_through']==now
                for e in r['negative_indices']:
                    age=round((now-float(s['times'][e]))*1000,9)
                    assert e>=127 and e<count and 125<=age<253
                    assert s['valid'][e] and s['y'][e]==0
                    mature_uses+=1
                updates[arm]['steps']+=r['gradient_steps']
                updates[arm]['nonzero']+=int(r['parameter_delta_norm']>0)
                updates[arm]['loss_decreased']+=int(r.get('loss_after',0)<r.get('loss_before',0))
                updates[arm]['clipped']+=int(r.get('gradient_norm',0)>1.)
    for key,c in counts.items():
        assert all(rows[key][k]==v for k,v in c.items()),key
        assert rows[key]['tpr']==c['tp']/180 and rows[key]['fpr']==c['fp']/180
    flips={}
    for level in ('calibrated','source_threshold'):
        for before,after in [('static',a) for a in MODES]+[('positive_replay','matured_plus_replay'),('real_forecast','matured_plus_replay')]:
            key=before+' -> '+after+'/'+level;changes=[]
            for sid in p['shot_ids']:
                a=byid[before+'/'+level][sid];b=byid[after+'/'+level][sid]
                if correct(a)!=correct(b):changes.append(dict(shot=sid,beneficial=correct(b),before=a,after=b))
            flips[key]=changes
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent()
    model=models_for_calibration(parent)['mapped_fixed'];initial=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');replay,manifest=historical_replay(p['preparation_ids'],mapping)
    assert manifest==p['replay_manifest']
    def deny(*args,**kwargs):raise AssertionError('no current/future outcome access during replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in (62765,87537):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=dict(transform({k:s[k] for k in ('values','available')},mapping),times=s['times'])
            curves,rs=predict_conditional(model,signal,replay)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for arm,value in curves.items():np.testing.assert_array_equal(value,z[arm])
            assert rs==READ(root/'shots'/f'{sid}_updates.json')
    assert_unchanged(model,initial);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_RECOUNT_MATURITY_AND_TWO_REPLAYS',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),shot_arm_checks=checks,negative_target_uses_checked=mature_uses,
        counts=counts,flips=flips,updates=updates,replayed_shots=[62765,87537],source_unchanged=True,
        immutable_early_false_alarms=early_fp,
        warning='Conditional non-event status assumption remains, and development-exposed data; not independent validation.')
    _durable_json(root/'verification.json',result)
    print({k:v for k,v in result.items() if k!='flips'},flush=True)
    print({k:dict(beneficial=sum(r['beneficial'] for r in v),harmful=sum(not r['beneficial'] for r in v)) for k,v in flips.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
