"""Independent aggregation, linear program refit, and fixed-shot GPU replay."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.optimize import linprog

from .. import io23
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ARMS,collect_shot,predict,eligible


def main(root):
    done=READ(root/'completed.json');assert done['status']=='completed'
    for file,digest in done['artifact_sha256'].items():assert sha(root/file)==digest,file
    protocol=READ(root/'protocol.json');policy=READ(root/'policy.json');fits=READ(root/'fit.json')
    rows=[r for sid in protocol['preparation_ids'] for r in READ(root/'geometry'/f'{sid}.json')['rows']]
    assert len(rows)==done['historical_rows']
    for arm,key in (('block','a'),('point','point_a')):
        arrays={}
        for field in (key,'b'):
            grouped=[]
            for group in ('safe','early','danger'):
                shots={}
                for r in rows:
                    if r['group']==group:shots.setdefault(r['shot'],[]).append(r[field])
                grouped.append(sum(np.sum(v,axis=0)/len(v) for v in shots.values())/len(shots))
            arrays[field]=np.array(grouped)
        a,b=arrays[key],arrays['b'];w=np.array(policy[arm]);f=fits[arm]
        np.testing.assert_allclose(a,f['risk_coefficients'],atol=1e-14,rtol=1e-12)
        np.testing.assert_allclose(b,f['signal_coefficients'],atol=1e-14,rtol=1e-12)
        assert (w>=0).all() and abs(w.sum()-1)<1e-8
        assert (b@w).min()>=-1e-8
        np.testing.assert_allclose((a@w).min(),f['margin'],atol=1e-10,rtol=1e-8)
        # Independent assembly of the maximin linear objective and constraints.
        A=np.zeros((6,24));A[:3,:23]=-a;A[:3,23]=1;A[3:,:23]=-b
        obj=np.zeros(24);obj[-1]=-1;eq=np.ones((1,24));eq[0,-1]=0
        solution=linprog(obj,A_ub=A,b_ub=np.zeros(6),A_eq=eq,b_eq=[1],
                         bounds=[(0,None)]*23+[(None,None)],method='highs-ds')
        assert solution.success
        np.testing.assert_allclose(-solution.fun,f['margin'],atol=1e-10,rtol=1e-8)
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');replays=[]
    def deny(*args,**kwargs):raise AssertionError('audit uses archived labels only')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in (54314,63226):
            old=READ(root/'geometry'/f'{sid}.json');raw=signal(sid)
            mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            replay,events=collect_shot(model,mapped,raw['times'],sid,old['is_disrupt'],old['down'])
            assert replay==old['rows'];assert events==old['events']
            replays.append(dict(shot=sid,kind='historical_geometry',rows=len(replay)))
        if done['results'] is not None:
            for sid in (89272,89284,89584):
                raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
                curves,updates=predict(model,mapped,policy)
                with np.load(root/'shots'/f'{sid}.npz') as z:
                    for arm in ('off',)+ARMS:np.testing.assert_array_equal(curves[arm],z[arm])
                assert updates==READ(root/'updates'/f'{sid}.json')
                replays.append(dict(shot=sid,kind='complete_stream',arms=len(curves)))
    metrics={};counts={}
    if done['results'] is not None:
        outcomes={r['shot']:r for r in READ(root/'outcomes.json')}
        decisions=READ(root/'per_shot.json')
        for arm in ('off',)+ARMS:
            tp=fp=early=0;danger=[];safe=[]
            for sid in protocol['shot_ids']:
                o=outcomes[sid]
                with np.load(root/'shots'/f'{sid}.npz') as z:
                    lead=np.round((o['down']-z['times'])*1000,9) if o['is_disrupt'] else None
                    valid=lead>=-1e-6 if o['is_disrupt'] else np.ones(len(z['times']),bool)
                    score=z[arm][valid].astype(np.float64);alarms=np.flatnonzero(score>=protocol['threshold'])
                    record=next(r for r in decisions[arm]['predictions'] if r['shot']==sid)
                    assert record['alarm']==bool(len(alarms))
                    if o['is_disrupt']:
                        timely=bool(len(alarms) and lead[valid][alarms[0]]>30.)
                        assert timely==record['timely'];tp+=int(timely)
                        early+=int(timely and lead[valid][alarms[0]]>3000.)
                        danger.append(float(-np.log(score[(lead[valid]>30)&(lead[valid]<=125)].clip(1e-12,1)).mean()))
                    else:
                        fp+=int(bool(len(alarms)));safe.append(float(-np.log1p(-score.max().clip(0,1-1e-12))))
            metrics[arm]=dict(tp=tp,fp=fp,very_early_over3000=early,danger_loss=float(np.mean(danger)),
                safe_loss=float(np.mean(safe)),balanced_loss=float((np.mean(danger)+np.mean(safe))/2))
            for k,v in metrics[arm].items():np.testing.assert_allclose(v,done['results'][arm][k],atol=1e-12,rtol=1e-10)
        for sid in protocol['shot_ids']:
            updates=READ(root/'updates'/f'{sid}.json')
            for arm,rs in updates.items():
                counts[arm]=counts.get(arm,0)+sum(r['gradient_steps'] for r in rs)
                for i,r in enumerate(rs):
                    assert r['last_observation']+1==r['first_affected_prediction']==256+128*i
                    assert r['actual_step_norm']<=.01001
                    if arm not in ('block','point') and not r['control_zero_direction']:
                        np.testing.assert_allclose(r['requested_step_norm'],r['main_requested_norm'],atol=1e-9,rtol=1e-6)
        assert eligible(done['results'])==done['eligible_for360']
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='passed',completed_sha256=sha(root/'completed.json'),historical_rows=len(rows),
        independent_LP_refits=2,replays=replays,independent_metrics=metrics,gradient_updates=counts,
        source_and_parent_unchanged=True,audit_code_sha256=sha(Path(__file__)))
    _durable_json(root/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
