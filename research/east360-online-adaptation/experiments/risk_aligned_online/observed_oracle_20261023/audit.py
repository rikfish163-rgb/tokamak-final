"""Recount every oracle curve, audit supervision timing, replay fixed shots."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..contextual_feedback_20261023.run import read_signal
from ..continual_episode_20261023.audit import recount
from ..continual_episode_20261023.run import labeled
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..heldout_threshold_20261023.run import REFERENCE,models_for_calibration
from ..history_matched_rule_20261023.probe import PREVIOUS
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import predict
from .run import PLAN


def correct(row):
    return bool(row['timely'] if row['is_disrupt'] else not row['alarm'])


def main(root,replay=False):
    torch.set_num_threads(2);torch.manual_seed(131)
    p=READ(root/'protocol.json');done=READ(root/'completed.json');stored=READ(root/'per_shot.json')
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    for f,h in p['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    for f,h in p['input_sha256'].items():assert sha(f)==h,f
    # Reused model/data modules were already captured by the parent experiment.
    # Check them too: the new launch snapshot contains only the new entrypoints.
    prior_protocol=READ(PREVIOUS/'protocol.json')
    for f,h in prior_protocol['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    assert sha(PLAN)==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256
    out={r['shot']:r for r in READ(PREVIOUS/'outcomes.json')}
    receipts=READ(root/'updates.json');assert [r['shot'] for r in receipts]==p['shot_ids']
    old=READ(PREVIOUS/'per_shot.json')['static/calibrated']['predictions']
    assert stored['static']['predictions']==old[:len(p['shot_ids'])]
    changes={a:dict(good=[],bad=[]) for a in ('oracle1','oracle4')}
    counters={a:dict(tp=0,fp=0,over3000=0,steps=0,training_loss_lower=0,training_loss_not_lower=0) for a in p['arms']}
    exposure=[];early_fp=[];checks=0
    for i,sid in enumerate(p['shot_ids']):
        o=out[sid];updates=receipts[i]['updates']
        with np.load(root/'shots'/f'{sid}.npz') as z:
            times=z['times'];flag=o['is_disrupt'];down=o['down']
            _,valid=io23.labels_at(times,flag,down,125.)
            np.testing.assert_array_equal(z['valid'],valid)
            for a in p['arms']:
                r=dict(shot=sid,**recount(times,z[a],flag,down,p['threshold']))
                assert r==stored[a]['predictions'][i],(sid,a);checks+=1
                counters[a]['tp']+=int(r['timely']);counters[a]['fp']+=int(not flag and r['alarm'])
                counters[a]['over3000']+=int(r['timely'] and r['lead_ms']>3000)
                if a!='static':
                    before=stored['static']['predictions'][i]
                    if correct(before)!=correct(r):changes[a]['good' if correct(r) else 'bad'].append(dict(shot=sid,before=before,after=r))
            srow=stored['static']['predictions'][i]
            if not flag and srow['alarm']:
                first=int(np.flatnonzero(z['static']>=p['threshold'])[0])
                if first<256:early_fp.append(dict(shot=sid,first=first))
            for a in ('oracle1','oracle4'):
                for r in updates[a]:
                    count=r['first_affected_prediction'];ends=np.asarray(r['supervised_ends'],dtype=int)
                    assert r['last_observation']==count-1 and count>=256 and count%128==0 and count<len(times)
                    expected=np.arange(count-113,count,16);expected=expected[valid[expected]]
                    np.testing.assert_array_equal(ends,expected)
                    counters[a]['steps']+=r['gradient_steps']
                    if r['gradient_steps']:
                        counters[a]['training_loss_lower' if r['loss_after']<r['loss_before'] else 'training_loss_not_lower']+=1
            if flag:
                positive=[]
                for r in updates['oracle1']:
                    count=r['first_affected_prediction'];ends=np.asarray(r['supervised_ends'],dtype=int)
                    # At least one selected positive target while the next output
                    # still has strictly more than30ms to the disruption.
                    next_lead=round((down-times[count])*1000.,9)
                    target_leads=np.round((down-times[ends])*1000.,9)
                    if next_lead>30 and ((target_leads<=125.)&(target_leads>=0)).any():positive.append(count)
                exposure.append(dict(shot=sid,static_timely=srow['timely'],timely_positive_update_counts=positive))
    for a,c in counters.items():
        assert c['tp']==done['metrics'][a]['tp'] and c['fp']==done['metrics'][a]['fp']
    replay_ids=[]
    if replay:
        parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];state=state_cpu(model)
        mapping=READ(REFERENCE/'input_map.json')
        for i in sorted({0,len(p['shot_ids'])//2,len(p['shot_ids'])-1}):
            sid=p['shot_ids'][i];s=labeled(read_signal(sid),io23.read_outcome('EAST',sid),mapping)
            curves,actual=predict(model,{k:s[k] for k in ('values','available')},s['y'],s['valid'])
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for a in curves:np.testing.assert_array_equal(curves[a],z[a])
            assert actual==receipts[i]['updates'];replay_ids.append(sid)
        assert_unchanged(model,state)
    summary=dict(positive_update_eligible=sum(bool(r['timely_positive_update_counts']) for r in exposure),
        no_timely_positive_update=sum(not r['timely_positive_update_counts'] for r in exposure),
        static_misses_without_positive_update=sum(not r['static_timely'] and not r['timely_positive_update_counts'] for r in exposure),
        early_fp=len(early_fp),changes={a:{k:len(v) for k,v in groups.items()} for a,groups in changes.items()})
    _durable_json(root/'verification.json',dict(status='VERIFIED_NONDEPLOYABLE_DIAGNOSTIC',
        completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        reference_code_hashes_verified=len(prior_protocol['code_sha256']),
        recounted_records=checks,replayed_shots=replay_ids,counts=counters,summary=summary,
        changes=changes,positive_exposure=exposure,early_fp=early_fp))
    print(dict(counts=counters,summary=summary,replayed_shots=replay_ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--replay',action='store_true')
    args=p.parse_args();main(args.output,args.replay)
