"""Independent counting and exact causal replay for the trajectory experiment."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ARMS,predict
from .run import READ,make_parent,models_for_calibration,REFERENCE,PREVIOUS,signal,transform,gradient_check


def main(root,replay=True):
    done=READ(root/'completed.json');p=READ(root/'protocol.json')
    assert len(p['shot_ids'])==360 and set(p['shot_ids'])==set(io23.cohort_ids('EAST'))
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    for f,h in READ(root/'prediction_seals.json').items():assert sha(root/f)==h,f
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==p['parent_sha256']
    assert sha(REFERENCE/'input_map.json')==p['mapping_sha256']
    bank_info=READ(root/'bank.json');truth=READ(root/'outcomes.json');rows=READ(root/'per_shot.json')
    assert [r['shot'] for r in truth]==p['shot_ids']
    with np.load(root/'bank.npz') as z:bank={k:z[k].copy() for k in z.files}
    bank.update(rows=bank_info['rows'],temperature=bank_info['temperature'])
    assert set(bank['shots'])==set(p['preparation_ids'])
    assert not set(bank['shots'])&set(p['shot_ids']+p['calibration_ids'])
    counts={a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0) for a in rows}
    updates={a:0 for a in ARMS};checks=0;weights_checked=0;all_details=[]
    for i,outcome in enumerate(truth):
        sid=outcome['shot'];flag=outcome['is_disrupt'];down=outcome['down']
        with np.load(root/'shots'/f'{sid}.npz') as z:
            times=z['times'];valid=np.ones(len(times),bool) if not flag else (down-times)*1000>=-1e-6
            for a in counts:
                hit=np.flatnonzero(z[a][valid]>=p['threshold']);alarm=bool(len(hit))
                lead=float(round((down-times[valid][hit[0]])*1000,9)) if flag and alarm else None
                timely=lead is not None and lead>30.
                record=dict(shot=sid,is_disrupt=flag,alarm=alarm,lead_ms=lead,timely=timely)
                assert record==rows[a]['predictions'][i],(sid,a)
                counts[a]['tp']+=int(timely);counts[a]['fp']+=int(alarm and not flag)
                counts[a]['n_disrupt']+=int(flag);counts[a]['n_safe']+=int(not flag);checks+=1
            receipts=READ(root/'receipts'/f'{sid}.json')
            assert len(receipts)==(len(times)-1)//128
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as old:np.testing.assert_array_equal(z['off'],old['static'])
            for j,r in enumerate(receipts):
                for a in ARMS:
                    v=r[a];ix=np.array(v['selected'],int);w=np.asarray(v['weights'])
                    assert v['last_observation']==(j+1)*128-1 and v['first_affected_prediction']==(j+1)*128
                    if not len(ix):
                        assert v['step_norm']==0 and v['reason']=='insufficient_joint_observations'
                        continue
                    assert len(set(bank['shots'][ix]))==len(ix)
                    yy=bank['labels'][ix];prior=np.array([1./np.sum(yy==y) for y in yy])
                    logits=-np.array(v['distances'])/bank['temperature']+np.log(prior)
                    expected=np.exp(logits-logits.max());expected/=expected.sum()
                    np.testing.assert_allclose(w,expected,atol=1e-14,rtol=1e-14)
                    assert v['requested_step_norm']==r['trajectory']['requested_step_norm']
                    assert abs(v['step_norm']-r['trajectory']['step_norm'])<2e-6
                    assert v['step_norm']<=.010002
                    updates[a]+=int(v['step_norm']>0);weights_checked+=1
            # Post-hoc diagnostic only: the last usable update before the
            # strongest timely/off endpoint. No labels enter the predictor.
            if flag:
                ix=np.flatnonzero(valid & (np.round((down-times)*1000,9)>30))
                if len(ix):
                    peak=int(ix[np.argmax(z['off'][ix])]);j=peak//128-1
                    detail=dict(shot=sid,parent_timely=rows['off']['predictions'][i]['timely'],end=peak,
                                parent_score=float(z['off'][peak]),scores={a:float(z[a][peak]) for a in ARMS})
                    if j>=0:
                        detail['positive_history_mass']={a:float(np.array(receipts[j][a]['weights'])@
                            bank['labels'][receipts[j][a]['selected']]) for a in ARMS}
                    all_details.append(detail)
    for a,c in counts.items():
        assert c['n_disrupt']==180 and c['n_safe']==180
        for k,v in c.items():assert done['metrics'][a][k]==v,(a,k)
        assert done['metrics'][a]['tpr']==c['tp']/180 and done['metrics'][a]['fpr']==c['fp']/180
    assert counts['off']['tp']==132 and counts['off']['fp']==37
    assert updates==done['temporary_updates']
    replays=[];check=None
    if replay:
        torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(True)
        parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];before=state_cpu(model)
        bank['gradients']=torch.load(root/'gradients.pt',map_location='cuda',weights_only=True)['gradients']
        check=gradient_check(model,bank);mapping=READ(REFERENCE/'input_map.json')
        def deny(*a,**k):raise AssertionError('no outcome access in replay')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in (54157,60999,74990,86155,96375):
                raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
                curves,receipts=predict(model,mapped,bank)
                with np.load(root/'shots'/f'{sid}.npz') as z:
                    for a,v in curves.items():np.testing.assert_array_equal(v,z[a])
                assert receipts==READ(root/'receipts'/f'{sid}.json')
                replays.append(sid);print(dict(replay=sid,status='exact'),flush=True)
        assert_unchanged(model,before)
    _durable_json(root/'routing_diagnostic.json',all_details)
    result=dict(completed_sha256=sha(root/'completed.json'),shot_decisions_recounted=checks,
                weight_vectors_recomputed=weights_checked,temporary_updates=updates,counts=counts,
                gradient_check=check,exact_replay_ids=replays,original_static_curves='all360 exact',
                scope='artifact arithmetic, causal replay and parent preservation; not efficacy certification')
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--no-replay',action='store_true')
    args=parser.parse_args();main(args.out,not args.no_replay)
