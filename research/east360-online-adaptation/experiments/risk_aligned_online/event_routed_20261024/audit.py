"""Independent counts, routing arithmetic and exact causal GPU replay."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from ..trajectory_memory_20261024.run import gradient_check
from .run import READ,PREVIOUS,REFERENCE,make_parent,models_for_calibration,signal,transform,state_cpu,assert_unchanged,load_metric_bank
from .learning import ARMS,predict


def describe(x):
    x=np.array(x,float)
    if not len(x):return dict(n=0)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),min=float(x.min()),max=float(x.max()))


def main(root):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(True)
    done=READ(root/'completed.json');p=READ(root/'protocol.json')
    assert set(p['shot_ids'])==set(io23.cohort_ids('EAST')) and len(p['shot_ids'])==360
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    for f,h in READ(root/'prediction_seals.json').items():assert sha(root/f)==h,f
    assert not set(p['shot_ids'])&set(p['preparation_ids']+p['calibration_ids'])
    bank=load_metric_bank()
    assert bank['metric_temperature']==READ(root/'metric_bank.json')['temperature']
    with np.load(root/'metric_bank.npz') as z:
        for mode,v in bank['metric_bank'].items():np.testing.assert_array_equal(v,z[mode])
    truth=READ(root/'outcomes.json');rows=READ(root/'per_shot.json')
    counts={a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0) for a in rows}
    updates={a:0 for a in ARMS};peaks={a:[] for a in rows};checks=0;weight_checks=0
    groups=('safe','early_disrupt','timely_danger','late_or_after_disruption')
    mass={a:{g:[] for g in groups} for a in ARMS}
    cosine={a:[] for a in ARMS if a!='risk'}
    for i,outcome in enumerate(truth):
        sid=outcome['shot'];flag=outcome['is_disrupt'];down=outcome['down'];assert sid==p['shot_ids'][i]
        with np.load(root/'shots'/f'{sid}.npz') as z:
            times=z['times'];valid=np.ones(len(times),bool) if not flag else (down-times)*1000>=-1e-6
            timely_valid=np.ones(len(times),bool) if not flag else np.round((down-times)*1000,9)>30
            assert timely_valid.any()
            for a in counts:
                hit=np.flatnonzero(z[a][valid]>=p['threshold']);alarm=bool(len(hit))
                lead=float(round((down-times[valid][hit[0]])*1000,9)) if flag and alarm else None
                timely=lead is not None and lead>30
                assert dict(shot=sid,is_disrupt=flag,alarm=alarm,lead_ms=lead,timely=timely)==rows[a]['predictions'][i]
                counts[a]['tp']+=int(timely);counts[a]['fp']+=int(alarm and not flag)
                counts[a]['n_disrupt']+=int(flag);counts[a]['n_safe']+=int(not flag)
                peaks[a].append(float(z[a][timely_valid].max()));checks+=1
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as old:np.testing.assert_array_equal(z['off'],old['off'])
        receipts=READ(root/'receipts'/f'{sid}.json');assert len(receipts)==(len(times)-1)//128
        for j,r in enumerate(receipts):
            lead=None if not flag else round((down-times[(j+1)*128-1])*1000,9)
            group='safe' if not flag else 'early_disrupt' if lead>125 else 'timely_danger' if lead>30 else 'late_or_after_disruption'
            mixed={}
            for a in ARMS:
                v=r[a];ix=np.array(v['selected'],int);w=np.array(v['weights'])
                assert v['last_observation']==(j+1)*128-1 and v['first_affected_prediction']==(j+1)*128
                assert v['step_norm']<=.010002
                assert abs(v['step_norm']-r['risk']['step_norm'])<2e-6
                assert v['requested_step_norm']==r['risk']['requested_step_norm']
                if not len(ix):
                    assert v['step_norm']==0 and v['reason']=='fewer_than8_observed_channels'
                    continue
                assert len(set(bank['shots'][ix]))==len(ix)
                yy=bank['labels'][ix];prior=np.array([1./np.sum(yy==y) for y in yy])
                mode='risk' if a in ('reverse','uniform') else a
                logw=-np.array(v['distances'])/bank['metric_temperature'][mode]+np.log(prior)
                expected=np.exp(logw-logw.max());expected/=expected.sum()
                np.testing.assert_allclose(w,expected,atol=1e-14,rtol=1e-14)
                updates[a]+=int(v['step_norm']>0);weight_checks+=1;mass[a][group].append(float(w[yy==1].sum()))
                if j in {0,len(receipts)//2,len(receipts)-1}:
                    mixed[a]=torch.as_tensor(w,device='cuda',dtype=torch.float32)@bank['gradients'][ix]
            if len(mixed)==len(ARMS):
                base=mixed['risk'].double()
                for a in cosine:
                    v=mixed[a].double();cosine[a].append(float((base@v)/(base.norm()*v.norm())))
    assert updates==done['temporary_updates']
    for a,cc in counts.items():
        for k,v in cc.items():assert done['metrics'][a][k]==v
        assert cc['n_disrupt']==cc['n_safe']==180
        assert done['metrics'][a]['tpr']==cc['tp']/180 and done['metrics'][a]['fpr']==cc['fp']/180
    assert counts['off']['tp']==132 and counts['off']['fp']==37
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];before=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');replays=[]
    def deny(*args,**kwargs):raise AssertionError('no outcome access during replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        gradcheck=gradient_check(model,bank)
        for sid in (54157,60999,74990,86155,96375):
            raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,receipts=predict(model,mapped,bank)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for a,v in curves.items():np.testing.assert_array_equal(v,z[a])
            assert receipts==READ(root/'receipts'/f'{sid}.json')
            replays.append(sid);print(dict(replay=sid,status='exact'),flush=True)
    assert_unchanged(model,before)
    yy=np.array([r['is_disrupt'] for r in truth]);ranking={}
    for a,values in peaks.items():
        values=np.array(values);diff=values[yy,None]-values[~yy][None,:]
        ranking[a]=float(np.mean((diff>0)+.5*(diff==0)))
    flips=READ(root/'flips.json');flipcounts={}
    for a,rr in flips.items():
        flipcounts[a]={name:sum(r['before']['is_disrupt']==d and r['beneficial']==b for r in rr)
                      for name,d,b in [('rescued',True,True),('lost',True,False),('removed_fp',False,True),('added_fp',False,False)]}
    diagnostic=dict(scope='post-hoc development mechanism diagnosis, not physical identification',
        timely_peak_auc=ranking,flips=flipcounts,direction_cosine_to_risk={a:describe(v) for a,v in cosine.items()},
        disruptive_history_mass={a:{g:describe(v) for g,v in gg.items()} for a,gg in mass.items()})
    _durable_json(root/'mechanism_diagnostic.json',diagnostic)
    verification=dict(completed_sha256=sha(root/'completed.json'),decisions_recounted=checks,
        weight_vectors_recomputed=weight_checks,counts=counts,exact_replay_ids=replays,
        gradient_check=gradcheck,all360_static_curves='exact old parent',source_and_parent_unchanged=True,
        scope='arithmetic/replay verification; not scientific efficacy certification')
    _durable_json(root/'verification.json',verification);print(verification,flush=True);print(diagnostic,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
