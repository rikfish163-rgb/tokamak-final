"""Matched-dose EAST intra-shot feedback based on cross-diagnostic order."""

import argparse
from pathlib import Path
from shutil import copyfile
from statistics import median
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_feedback_precision_20261113 as parent
from . import commissioning_history_span_20261114 as history
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_peak_feedback_20261114 import confirmed
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips


group, lag = parent.group, parent.lag
CHANNELS = tuple(io23.RAW23[lag.CHANNELS[i]] for i in group.USE)
GROUPS = group.SIGNED_GROUPS
# One pair per physical question of ordering. The grouping limits
# interpretation, not membership of the 23-channel classifier.
PAIRS = tuple((CHANNELS.index(a), CHANNELS.index(b)) for a,b in (
    ('n=1 amplitude','sxr_var'), ('n=1 amplitude','xuv_var'),
    ('dx_a','sxr_skew'), ('dy_a','xuv_skew'), ('v_loop','sxr_var'), ('v_loop','xuv_var')))
BASES = ('fixed','bias_only','current_5','earlier_5')
TRIALS = ('order_now','order_reverse','order_earlier')
ARMS = (*BASES,*TRIALS)
CASES = (70320,73405,74970,85422,76956)
OFFSET = lag.H+1+parent.WAIT


def path_area(x):
    """Twice the signed enclosed area, closed by a straight endpoint chord."""
    z = x-x[:,:1]
    i,j = zip(*PAIRS,strict=True)
    return (z[:,:-1,i]*z[:,1:,j]-z[:,:-1,j]*z[:,1:,i]).sum(1)


def area_view(x):
    # Retain sign and damp amplitude in the same pre-issue standardized chart.
    return path_area(x).asinh()


def older_indices(q, observed):
    eligible = np.maximum.accumulate(np.r_[-1,np.where(observed,np.arange(len(q)),-1)])
    i = eligible[np.searchsorted(q+OFFSET,q,side='right')]
    return i,observed & (i>=0)


def rotate(origin, now, previous, strength):
    """Change only the coefficient component orthogonal to the legacy write."""
    a,b,c = (v.flatten(1) for v in (origin,now,previous))
    n = a.norm(dim=-1)
    first = F.normalize(a,dim=-1,eps=group.EPS)

    def orthogonal(x):
        drop = x-(x*first).sum(-1,keepdim=True)*first
        return F.normalize(drop,dim=-1,eps=group.EPS),drop.norm(dim=-1)

    u,un = orthogonal(b)
    v,vn = orthogonal(c)
    live = (un>1e-7) & (vn>1e-7) & (n>1e-7)
    r = torch.where(live,strength,0.)[:,None]
    s = (1.-r.square()).sqrt()*first
    result = [((s+r*g)*n[:,None]).reshape_as(origin) for g in (u,-u,v)]
    return torch.stack([torch.where(live[:,None,None],d,origin) for d in result])


def track(signal,args,decoder,path,teacher,archived):
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    base = args[0]
    eligible,fields,truth = group.feedback_waveform(signal,q,args[2],z.device)
    # The common legacy path is untouched. New order terms additionally need
    # actual measurements at every point, not imputed diagnostic values.
    columns = [lag.CHANNELS[k] for k in group.USE]
    at = q[:,None]+np.arange(1,lag.H+1)
    exact = signal['available'][at][...,columns].all(axis=(1,2))
    now = area_view(group.signed_levels(truth,fields[1],fields[2]))
    old,paired = older_indices(q,eligible & exact)
    # Compare earlier increments in that block's own chart: converting the
    # entire earlier waveform to the current chart would confound the control.
    old_area = now[old.clip(min=0)]
    pending = prior.new_zeros((len(q),lag.H//group.BIN,len(group.USE)))
    _,_,_,common = group.past_force_parts(q,pending,eligible)
    # The sealed archive stores eligibility of the legacy *innovation*,
    # not of an individual waveform: it also needs earlier observed packets.
    np.testing.assert_array_equal(common,archived)
    states = prior.new_zeros((len(TRIALS)+1,len(q)+1,base.cfg.window//base.cfg.patch-1,len(group.TRUSTED)))
    geometry = np.zeros((len(q),5),np.float32)
    supported = np.flatnonzero(eligible)
    for block in range(0,len(supported),group.BATCH):
        i = supported[block:block+group.BATCH]
        issued,changed,state,wave,center,spread = group.signed_window(
            base,signal,q,i,phase,correction,prior,fields,decoder)
        pred,pending[i] = parent.signed_residual(wave,truth[i],center,spread)
        _,use,_,_ = group.past_force_parts(q,pending,eligible)
        accepted = torch.as_tensor(common[i],device=z.device)
        d = group.shape_force(use[i],teacher)*accepted[:,None,None]
        measured = now[i]
        valid = torch.as_tensor((common & paired)[i],device=z.device)
        forecast = area_view(pred)
        before = old_area[i]
        size = torch.minimum(measured.square().mean(1).sqrt(),before.square().mean(1).sqrt())
        strength = .5*size.tanh()*valid
        # Difference in squared error to +area and -area cancels all terms
        # shared by the targets. There is no unsigned surprise component.
        gg,u,v = group.signed_directions(issued,changed,state,
            ((pred,d),(forecast,measured*valid[:,None]),(forecast,before*valid[:,None])))
        edits = rotate(gg,u,v,strength)
        states[:,i+1] = torch.cat((gg[None],edits))
        lv = F.cosine_similarity(edits[0].flatten(1),edits[1].flatten(1))
        geometry[i] = torch.stack((valid,measured.abs().mean(1),before.abs().mean(1),
                                   forecast.detach().abs().mean(1),lv),dim=1).detach().cpu().numpy()
    if not torch.isfinite(states).all() or not np.isfinite(geometry).all():
        raise FloatingPointError('nonfinite cross-diagnostic write')
    return (q,h,z,beta,phase,correction,states,common),geometry


def replay(out):
    begun = time.monotonic()
    lock,p,s,args,decoder,path = history.inputs()
    base = args[0]
    teacher = torch.as_tensor(read(parent.delay.HISTORY/'teaching.json')['weights'],
                              device=base.center.device,dtype=base.center.dtype)
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(scope='EAST360_CAUSAL_CROSS_DIAGNOSTIC_PATH_ORDER_DEVELOPMENT',
        **{k:p[k] for k in history.LOCK}, reference=str(history.ARCHIVE), arms=ARMS,
        pairs=[[CHANNELS[i],CHANNELS[j]] for i,j in PAIRS], offset=OFFSET, handoff=base.cfg.patch,
        feedback='fixed old internal encoder; direct H25 prediction and actual five bins; train only the temporary'
                 ' state from signed measured area, its sign-reversed and its already-arrived predecessor',
        allocation='rotate an equal fraction of the original update into orthogonal directions; same actual token'
                   ' L2 and same phase; undefined or zero genuine/early direction leaves original untouched',
        outcomes='original HDF5 only after all 360 predictions sealed; original cold20 counted, no new inter-shot labels'))
    completed, witness = [],{}
    accuracy = np.zeros(3)
    theta = (*p['calibration'].values(),p['original'])
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('outcomes forbidden before seal')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('outcomes forbidden before seal'))):
        for rank,sid in enumerate(p['order'],1):
            key = f'{rank:04d}_{sid}.npz'
            a = history.archived_shot(rank,sid,s)
            n = len(a['dose'])
            curves = np.broadcast_to(a['current_5'],(1+len(TRIALS),n))
            geo = np.zeros((len(a['release']),5),np.float32)
            if rank>p['startup']:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('the bound signal file changed')
                x = lag.prepare_actual(simple_data.read_shot('EAST',sid,rate_contract='nominal'),rank,lock['mapping'])
                t,geo = track(x,args,decoder,path,teacher,a['eligible'])
                np.testing.assert_array_equal(t[0]+lag.H+1,a['release'])
                curves,pd,dd = parent.render(base,x,t,a['bias_only'],a['dose'])
                np.testing.assert_allclose(curves[0],a['current_5'],rtol=0.,atol=3e-6)
                for k in theta:
                    np.testing.assert_array_equal(curves[0]>=k,a['current_5']>=k)
                accuracy[:2] = np.maximum(accuracy[:2],[pd,dd])
                if rank in (21,120,360) or sid in CASES:
                    first = np.flatnonzero((curves[1]>=p['calibration']['0.2']) & (a['dose']>0))
                    cut = min(n,max(520,int(first[0])+2) if len(first) else 520)
                    xx = {k:x[k][:cut] for k in ('values','available')}
                    prior = a['release']<cut
                    tt,geo2 = track(xx,args,decoder,path,teacher,a['eligible'][prior])
                    yy,*_ = parent.render(base,xx,tt,a['bias_only'][:cut],a['dose'][:cut])
                    np.testing.assert_allclose(geo2,geo[prior],rtol=0.,atol=3e-6)
                    np.testing.assert_allclose(yy,curves[:,:cut],rtol=0.,atol=3e-6)
                    for k in theta:
                        np.testing.assert_array_equal(yy>=k,curves[:,:cut]>=k)
                    err = float(np.abs(yy-curves[:,:cut]).max())
                    accuracy[2] = max(accuracy[2],err)
                    witness[sid] = dict(points=cut,max_difference=err)
            with (out/'shots'/key).open('xb') as f:
                np.savez_compressed(f,current_control=curves[0], geometry=geo,
                                   **dict(zip(TRIALS,curves[1:],strict=True)))
            completed.append(key)
            if rank in (21,60,120,180,240,300,360):
                print(dict(stage='sealed_curves',rank=rank,seconds=time.monotonic()-begun),flush=True)
    write(out/'seal.json',dict(completed=completed,witness=witness, errors=accuracy.tolist(),
                              elapsed_s=time.monotonic()-begun))


def finish(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    expected = [f'{r:04d}_{sid}.npz' for r,sid in enumerate(p['order'],1)]
    if len(expected)!=360 or s['completed']!=expected:
        raise ValueError('all predictions must be sealed before scoring')
    old,ref = (read(history.ARCHIVE/f'{k}.json') for k in ('per_shot','seal'))
    scores = {b:{k:[] for k in ARMS} for b in p['calibration']}
    confirmation = {b:{n:{a:[0,0] for a in ARMS} for n in (2,3,5)} for b in scores}
    critical,writes,grads = [],0,[]
    for rank,(sid,key) in enumerate(zip(p['order'],expected,strict=True),1):
        d,down,t = group.hdf._raw_truth(sid)
        a = history.archived_shot(rank,sid,ref)
        with np.load(out/'shots'/key,allow_pickle=False) as f:
            curves = {k:(f if k in TRIALS else a)[k] for k in ARMS}
            active = a['dose']>0.
            for k in TRIALS:
                np.testing.assert_array_equal(curves[k][~active],a['current_5'][~active])
            writes += int(active.sum())
            grads.append(f['geometry'][f['geometry'][:,0]>0.])
            for b,tau in p['calibration'].items():
                cut = p['original'] if rank<=p['startup'] else tau
                for k,risk in curves.items():
                    row = dict(shot=sid,disruptive=d,down_s=down,**score_curve(t,risk,d,down,cut))
                    if k in BASES and row!=old[b][k][rank-1]:
                        raise ValueError('archive alarm no longer matches original HDF5')
                    scores[b][k].append(row)
                    for n,sub in confirmation[b].items():
                        alert,ms = confirmed(t,risk,down,cut,n)
                        sub[k][0] += bool(d and alert and ms>30.)
                        sub[k][1] += bool(not d and alert)
            if rank>p['startup'] and d and not scores['0.2']['current_5'][-1]['timely']:
                ms = np.round((down-t)*1e3,9)
                use = (ms>30.) & (ms<=125.) & active
                logits = {k:logit(curves[k][use].clip(1e-6,1-1e-6)) for k in ('current_5',*TRIALS)}
                p0 = logits['current_5']
                at = np.flatnonzero(use)
                j = np.searchsorted(a['release']+parent.WAIT,at,side='right')-1
                c = p['calibration']['0.2']
                critical.append(dict(shot=sid,kind='late' if scores['0.2']['current_5'][-1]['alarm'] else 'never',
                    endpoints=len(at),area_eligible=int(np.sum(f['geometry'][j,0]>0.)),
                    gap=float(logit(c)-p0.max()) if len(p0) else None,
                    max_lift={k:float((v-p0).max()) if len(p0) else None for k,v in logits.items() if k!='current_5'},
                    rescued={k:bool(np.any((curves[k][use]>=c) & (a['current_5'][use]<c))) for k in TRIALS}))
    metric = {b:{k:lead.summarize(v) for k,v in r.items()} for b,r in scores.items()}
    pairs = {b:{f'{a}__{k}':dict(flips=flips(v[a],v[k]),lead=lead.paired(v[a],v[k]))
                for a,k in (('bias_only',TRIALS[0]),('current_5',TRIALS[0]),
                            (TRIALS[1],TRIALS[0]),(TRIALS[2],TRIALS[0]))} for b,v in scores.items()}
    g = np.concatenate(grads)
    write(out/'per_shot.json',scores)
    write(out/'result.json',dict(metrics=metric,paired=pairs,confirmation=confirmation,
        critical=critical, writes=writes, paired_feedback=len(g),
        geometry_columns=['complete_pair','mean_absolute_observed_area','mean_absolute_earlier_area',
                          'mean_absolute_predicted_area','now_reverse_update_cosine'],
        geometry=np.quantile(g,[.1,.5,.9],axis=0).tolist(),
        median_near_window_gap=median(v['gap'] for v in critical if v['gap'] is not None),
        raw_verified=360*len(scores)*len(ARMS)))
    for b,r in metric.items():
        print(b,{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in r.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--score-only',action='store_true')
    a = parser.parse_args()
    target = a.output.resolve()
    if not a.score_only:
        replay(target)
    finish(target)
