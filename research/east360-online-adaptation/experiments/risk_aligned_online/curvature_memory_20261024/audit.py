"""Recount decisions, verify cached geometry and replay causal GPU predictions."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..trajectory_memory_20261024.diagnose import describe
from .learning import ARMS, predict, bernoulli_rows, solve_directions, functional_match
from .run import READ, PREVIOUS, REFERENCE, load_bank, make_parent, models_for_calibration, signal, transform


def main(root):
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(True)
    done=READ(root/'completed.json');p=READ(root/'protocol.json')
    assert len(p['shot_ids'])==360 and set(p['shot_ids'])==set(io23.cohort_ids('EAST'))
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    for f,h in READ(root/'prediction_seals.json').items():assert sha(root/f)==h,f
    for f,h in p['historical_bank_sha256'].items():assert sha(PREVIOUS/f)==h,f
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==p['parent_sha256']
    assert sha(REFERENCE/'input_map.json')==p['mapping_sha256']
    bank=load_bank();geometry=torch.load(root/'geometry.pt',map_location='cpu',weights_only=True)
    info=READ(root/'geometry.json')
    u=bernoulli_rows(geometry['jacobian'].double(),geometry['p'],geometry['q'])
    full,diagonal,independent=solve_directions(bank['gradients'].cpu().double(),u)
    for k,v in (('u',u),('full',full),('diagonal',diagonal)):
        torch.testing.assert_close(v.float(),geometry[k],atol=0,rtol=0)
        bank[k]=geometry[k].cuda()
    assert independent['damping']==info['damping']
    eigen=torch.linalg.eigvalsh(u@u.T).flip(0)
    if float(eigen.min()) < -1e-10*float(eigen.max()):raise ArithmeticError('non-PSD historical curvature')
    # Remove only numerical negative roundoff for an explicitly PSD Gram.
    eigen=eigen.clamp_min(0);mass=eigen/eigen.sum()
    spectrum=dict(top_eigenvalue_share=float(mass[0]),
        directions_for95pct=int(torch.searchsorted(mass.cumsum(0),.95))+1,
        directions_for99pct=int(torch.searchsorted(mass.cumsum(0),.99))+1,
        eigenvalues_above_damping=int((eigen>info['damping']).sum()),
        participation_ratio=float(eigen.sum().square()/eigen.square().sum()))
    assert set(bank['shots'])==set(p['preparation_ids'])
    assert not set(bank['shots'])&set(p['shot_ids']+p['calibration_ids'])
    truth=READ(root/'outcomes.json');rows=READ(root/'per_shot.json')
    counts={a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0) for a in rows}
    updates={a:0 for a in ARMS};checks=0;weights_checked=0;peaks={a:[] for a in rows}
    cosine={a:[] for a in ('raw','diagonal','level','reverse','uniform')}
    rawkl_ratio=[];kl_errors=[];direction_checks=0
    for i,outcome in enumerate(truth):
        sid=outcome['shot'];flag=outcome['is_disrupt'];down=outcome['down']
        assert sid==p['shot_ids'][i]
        with np.load(root/'shots'/f'{sid}.npz') as z:
            times=z['times'];valid=np.ones(len(times),bool) if not flag else (down-times)*1000>=-1e-6
            timely_valid=np.ones(len(times),bool) if not flag else np.round((down-times)*1000,9)>30
            if not timely_valid.any():raise ValueError('no timely observation; do not silently omit a shot')
            for a in counts:
                hit=np.flatnonzero(z[a][valid]>=p['threshold']);alarm=bool(len(hit))
                lead=float(round((down-times[valid][hit[0]])*1000,9)) if flag and alarm else None
                timely=lead is not None and lead>30.
                record=dict(shot=sid,is_disrupt=flag,alarm=alarm,lead_ms=lead,timely=timely)
                assert record==rows[a]['predictions'][i],(sid,a)
                counts[a]['tp']+=int(timely);counts[a]['fp']+=int(alarm and not flag)
                counts[a]['n_disrupt']+=int(flag);counts[a]['n_safe']+=int(not flag);checks+=1
                peaks[a].append(float(z[a][timely_valid].max()))
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as old:
                np.testing.assert_array_equal(z['off'],old['off'])
                np.testing.assert_array_equal(z['raw'],old['trajectory'])
        receipts=READ(root/'receipts'/f'{sid}.json');assert len(receipts)==(len(times)-1)//128
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
                if a!='raw_kl':
                    assert v['requested_step_norm']==r['raw']['requested_step_norm']
                    assert abs(v['step_norm']-r['raw']['step_norm'])<2e-6
                assert v['step_norm']<=.010002
                updates[a]+=int(v['step_norm']>0);weights_checked+=1
            # Predetermined first/middle/last write per shot, not selected by
            # outcomes. Verify the functional-match control actually matches.
            if j in {0,len(receipts)//2,len(receipts)-1} and all(r[a]['selected'] for a in ARMS):
                mixed={}
                for a in cosine.keys()|{'full'}:
                    v=r[a];kind='gradients' if a=='raw' else 'diagonal' if a=='diagonal' else 'full'
                    mixed[a]=torch.as_tensor(v['weights'],dtype=torch.float32,device='cuda')@bank[kind][v['selected']]
                length=r['full']['requested_step_norm']
                expected=functional_match(mixed['raw'],mixed['full'],bank['u'],length)
                assert abs(expected-r['raw_kl']['requested_step_norm'])<1e-14
                rawkl_ratio.append(expected/length if length else 0.)
                energy={}
                for a,ell in (('raw',expected),('full',length)):
                    d=mixed[a];step=d*(ell/float(d.double().norm()))
                    energy[a]=float((bank['u']@step).double().square().sum())
                relative=abs(energy['raw']-energy['full'])/max(energy.values())
                if relative>2e-5:raise AssertionError(('functional energy mismatch',sid,j,relative))
                kl_errors.append(relative)
                ref=mixed['full'].double()
                for a in cosine:
                    v=mixed[a].double();cosine[a].append(float((ref@v)/(ref.norm()*v.norm())))
                direction_checks+=1
    for a,c in counts.items():
        assert c['n_disrupt']==180 and c['n_safe']==180
        for k,v in c.items():assert done['metrics'][a][k]==v,(a,k)
        assert done['metrics'][a]['tpr']==c['tp']/180 and done['metrics'][a]['fpr']==c['fp']/180
    assert counts['off']['tp']==132 and counts['off']['fp']==37
    assert counts['raw']['tp']==143 and counts['raw']['fp']==58
    assert updates==done['temporary_updates']
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];before=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');replays=[]
    def deny(*a,**k):raise AssertionError('no outcome access during replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
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
        values=np.asarray(values,dtype='float32');safe=np.sort(values[~yy]);oracle={}
        d=values[yy,None]-values[~yy][None,:]
        auc=float(np.mean((d>0)+.5*(d==0)))
        for cap in (.1,.15,.2):
            allowed=int(round(cap*len(safe)))
            threshold=float(np.nextafter(safe[len(safe)-allowed-1],np.float32(np.inf)))
            oracle[str(cap)]=dict(threshold=threshold,tp=int(np.sum(values[yy]>=threshold)),fp=int(np.sum(values[~yy]>=threshold)))
        ranking[a]=dict(timely_peak_auc=auc,oracle_NOT_deployable=oracle)
    flips=READ(root/'flips.json');flip_counts={}
    for a,rr in flips.items():
        flip_counts[a]={name:sum(r['before']['is_disrupt']==d and r['beneficial']==b for r in rr)
                       for name,d,b in [('rescued',True,True),('lost',True,False),('removed_fp',False,True),('added_fp',False,False)]}
    paired={}
    for other in ('off',)+tuple(a for a in ARMS if a!='full'):
        gains=[];losses=[]
        for current,reference in zip(rows['full']['predictions'],rows[other]['predictions']):
            assert current['shot']==reference['shot']
            good=lambda r:bool(r['timely']) if r['is_disrupt'] else not r['alarm']
            if good(current) and not good(reference):gains.append(current['shot'])
            if good(reference) and not good(current):losses.append(current['shot'])
        paired[other]=dict(full_better=gains,full_worse=losses,net_correct=len(gains)-len(losses))
    diagnostic=dict(scope='post-hoc DEVELOPMENT_ONLY; test-derived thresholds NOT deployable or capability upper bounds',
        ranking=ranking,flips=flip_counts,curvature_spectrum=spectrum,paired_full_vs=paired,direction_checks=direction_checks,cosine_to_full={a:describe(v) for a,v in cosine.items()},
        raw_kl_length_ratio=describe(rawkl_ratio),functional_match_relative_error=describe(kl_errors))
    _durable_json(root/'mechanism_diagnostic.json',diagnostic)
    verification=dict(completed_sha256=sha(root/'completed.json'),shot_decisions_recounted=checks,
        weight_vectors_recomputed=weights_checked,temporary_updates=updates,counts=counts,
        geometry_double_solve='exact float32 cast reproduced',exact_replay_ids=replays,
        static_and_raw_curves='all360 both exact',functional_control_points=direction_checks,
        scope='arithmetic, causal replay and preservation checks; not efficacy certification')
    _durable_json(root/'verification.json',verification)
    print(verification,flush=True);print(diagnostic,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
