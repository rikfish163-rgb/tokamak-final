"""Recount predictions, reconstruct paired gradients, replay fixed GPU shots."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..event_routed_20261024.audit import describe
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ARMS, prepare_pairs, predict
from .run import READ, POINT, REFERENCE, make_parent, models_for_calibration, load_metric_bank, load_histories, signal, transform, state_cpu, assert_unchanged, DEFAULT_PARENT, PARENT_SHA256


def main(root):
    torch.set_num_threads(2); torch.backends.mha.set_fastpath_enabled(True)
    torch.backends.cuda.matmul.allow_tf32=False
    done=READ(root/'completed.json'); p=READ(root/'protocol.json')
    assert len(p['shot_ids'])==360 and set(p['shot_ids'])==set(io23.cohort_ids('EAST'))
    assert not set(p['shot_ids']) & set(p['preparation_ids']+p['calibration_ids'])
    for f,h in done['artifact_sha256'].items():
        assert sha(root/f)==h,f
    for f,h in READ(root/'prediction_seals.json').items():
        assert sha(root/f)==h,f
    for f,h in p['input_sha256'].items():
        assert sha(Path(f))==h,f
    bank=load_metric_bank(); parent,_=make_parent(); model=models_for_calibration(parent)['mapped_fixed']
    original=state_cpu(model); mapping=READ(REFERENCE/'input_map.json')
    pairs, rows=prepare_pairs(model, bank, load_histories(p['preparation_ids'],mapping))
    assert rows==READ(root/'pairs.json')
    with np.load(root/'pairs.npz') as z:
        for k,v in pairs.items():
            np.testing.assert_array_equal(v,z[k])
    saved=torch.load(root/'pair_gradients.pt',map_location='cuda',weights_only=True)
    for k,g in bank['pair_gradients'].items():
        torch.testing.assert_close(g,saved[k],atol=0,rtol=0)
    truth=READ(root/'outcomes.json'); decisions=READ(root/'per_shot.json')
    counts={a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0) for a in ('off',)+ARMS}
    peaks={a:[] for a in counts}; updates={a:0 for a in ARMS}
    cosine={a:[] for a in ARMS if a!='paired'}
    group_names=('safe','early_disrupt','timely_danger','late_or_after_disruption')
    changes={a:{g:[] for g in group_names} for a in ARMS}
    weight_checks=0; checked=0
    for i,outcome in enumerate(truth):
        sid=outcome['shot']; flag=outcome['is_disrupt']; down=outcome['down']
        assert sid==p['shot_ids'][i]
        with np.load(root/'shots'/f'{sid}.npz') as z:
            times=z['times']; valid=np.ones(len(times),bool) if not flag else (down-times)*1000>=-1e-6
            timely_valid=np.ones(len(times),bool) if not flag else np.round((down-times)*1000,9)>30
            for a in counts:
                hit=np.flatnonzero(z[a][valid]>=p['threshold']); alarm=bool(len(hit))
                lead=float(round((down-times[valid][hit[0]])*1000,9)) if flag and alarm else None
                timely=lead is not None and lead>30
                assert dict(shot=sid,is_disrupt=flag,alarm=alarm,lead_ms=lead,timely=timely)==decisions[a]['predictions'][i]
                counts[a]['tp']+=int(timely); counts[a]['fp']+=int(alarm and not flag)
                counts[a]['n_disrupt']+=int(flag); counts[a]['n_safe']+=int(not flag)
                peaks[a].append(float(z[a][timely_valid].max())); checked+=1
            with np.load(POINT/'shots'/f'{sid}.npz') as old:
                np.testing.assert_array_equal(z['off'],old['off'])
                np.testing.assert_array_equal(z['point'],old['risk'])
            receipts=READ(root/'receipts'/f'{sid}.json')
            assert len(receipts)==(len(times)-1)//128
            for j,r in enumerate(receipts):
                index=(j+1)*128
                lead=None if not flag else round((down-times[index])*1000,9)
                group='safe' if not flag else 'early_disrupt' if lead>125 else 'timely_danger' if lead>30 else 'late_or_after_disruption'
                mixed={}
                for a in ARMS:
                    v=r[a]; ix=np.array(v['selected'],int); w=np.array(v['weights'])
                    assert v['last_observation']==index-1 and v['first_affected_prediction']==index
                    assert v['step_norm']<=.010002 and abs(v['step_norm']-r['point']['step_norm'])<2e-6
                    assert v['requested_step_norm']==r['point']['requested_step_norm']
                    changes[a][group].append(float(z[a][index]-z['off'][index]))
                    if a in ('augmented','paired','unpaired'):
                        assert v['selected']==r['point']['selected'] and v['weights']==r['point']['weights']
                    if not len(ix):
                        assert v['step_norm']==0
                        continue
                    assert len(set(bank['shots'][ix]))==len(ix)
                    yy=bank['labels'][ix]; prior=np.array([1./np.sum(yy==y) for y in yy])
                    logw=-np.array(v['distances'])/bank['metric_temperature']['risk']+np.log(prior)
                    expected=np.exp(logw-logw.max()); expected/=expected.sum()
                    np.testing.assert_allclose(w,expected,atol=1e-14,rtol=1e-14)
                    updates[a]+=int(v['step_norm']>0); weight_checks+=1
                    if j in {0,len(receipts)//2,len(receipts)-1}:
                        g=bank['gradients'] if a=='point' else bank['pair_gradients'][a if a in ('augmented','unpaired') else 'paired']
                        mixed[a]=torch.as_tensor(w,device='cuda',dtype=torch.float32)@g[ix]
                if len(mixed)==len(ARMS):
                    base=mixed['paired'].double()
                    for a in cosine:
                        v=mixed[a].double()
                        cosine[a].append(float((base@v)/(base.norm()*v.norm())))
    assert updates==done['temporary_updates']
    for a,cc in counts.items():
        for k,v in cc.items():
            assert done['metrics'][a][k]==v
        assert cc['n_disrupt']==cc['n_safe']==180
        assert done['metrics'][a]['tpr']==cc['tp']/180 and done['metrics'][a]['fpr']==cc['fp']/180
    assert counts['off']['tp']==132 and counts['off']['fp']==37
    assert counts['point']['tp']==148 and counts['point']['fp']==70
    replays=[]
    def deny(*args,**kwargs):
        raise AssertionError('no outcomes during replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in (54157,60999,74990,86155,96375):
            raw=signal(sid); mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,receipts=predict(model,mapped,bank)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for a,v in curves.items():
                    np.testing.assert_array_equal(v,z[a])
            assert receipts==READ(root/'receipts'/f'{sid}.json')
            replays.append(sid); print(dict(replay=sid,status='exact'),flush=True)
    assert_unchanged(model,original); assert sha(DEFAULT_PARENT)==PARENT_SHA256
    yy=np.array([r['is_disrupt'] for r in truth]); ranking={}
    for a,values in peaks.items():
        values=np.array(values); diff=values[yy,None]-values[~yy][None,:]
        ranking[a]=float(np.mean((diff>0)+.5*(diff==0)))
    flips=READ(root/'flips.json'); flipcounts={}
    for a,rr in flips.items():
        flipcounts[a]={name:sum(r['before']['is_disrupt']==d and r['beneficial']==b for r in rr)
            for name,d,b in [('rescued',True,True),('lost',True,False),('removed_fp',False,True),('added_fp',False,False)]}
    history=[]
    for sid in sorted({r['shot'] for r in rows}):
        selected=[r for r in rows if r['shot']==sid]
        history.append(dict(shot=sid,pairs=len(selected),distinct_early=len({r['early_end'] for r in selected}),
            early_lead_ms=describe([r['early_lead_ms'] for r in selected]),
            early_probability=describe([r['parent']['paired']['early'] for r in selected]),
            late_probability=describe([r['parent']['paired']['late'] for r in selected])))
    diagnostic=dict(scope='post-hoc development diagnosis; first post-update samples are correlated, not independent evidence',
        timely_peak_auc=ranking,flips=flipcounts,history=history,
        direction_cosine_to_paired={a:describe(v) for a,v in cosine.items()},
        first_post_update_probability_change={a:{g:describe(v) for g,v in groups.items()} for a,groups in changes.items()})
    _durable_json(root/'mechanism_diagnostic.json',diagnostic)
    verification=dict(completed_sha256=sha(root/'completed.json'),decisions_recounted=checked,
        weight_vectors_recomputed=weight_checks,counts=counts,exact_replay_ids=replays,
        paired_gradients_recomputed='64 paired rows x3 objectives newly differentiated;256 safe rows per objective retained cached exact',
        historical_pairs_rebuilt=len(rows),audit_code_sha256=sha(Path(__file__)),
        all360_static_and_point='exact previous curves',source_and_parent_unchanged=True,
        scope='arithmetic and replay verification, not scientific certification')
    _durable_json(root/'verification.json',verification)
    print(verification,flush=True); print(diagnostic,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
