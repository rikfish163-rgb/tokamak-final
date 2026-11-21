"""Test current-conditioned voltage and radiation order on sealed EAST errors."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit

from . import commissioning_coincident_innovation_20261111 as joint
from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_excursion_order_20261111 as excursion
from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .commissioning_transition_source_20261110 import shot_coordinates
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write


REFERENCE = io23.ROOT/'probe_outputs/commissioning_signed_innovations_east360_20261112_v1'
PARENT = 'innovation_current'
COLS = tuple(io23.RAW23.index(k) for k in ('ip','Bt','v_loop','sxr_mean','xuv_ratio'))
BLOCK = joint.BLOCK_SHOTS
CALIPERS = np.r_[excursion.CALIPERS[:4], .35, .25, .25, .25]
PREDICTIONS = ('hold_voltage','voltage_history','current_conditioned','radiation_leads','radiation_older')
MODELS = ('marginal','raw_order','raw_shifted','conditional_order','conditional_shifted')
RIDGE = .10


def aligned_scores(voltage, radiation):
    """Signed adjacent-bin contrast; rotate observed radiation only after full arrival."""
    v = voltage-voltage.mean(1,keepdims=True)
    v /= np.maximum(np.linalg.norm(v,axis=1,keepdims=True), .05)
    rad = radiation-radiation.mean(1,keepdims=True)
    rad /= np.maximum(np.linalg.norm(rad,axis=1,keepdims=True), .05)

    def relation(x):
        return .5*((v[:,1:,None]*x[:,:-1]).sum(1) - (v[:,:-1,None]*x[:,1:]).sum(1))

    return relation(rad),relation(np.roll(rad,2,axis=1))


def _estimate(x, y, target):
    fit = x[:,4:25]
    ridge = np.eye(fit.shape[2])*RIDGE*fit.shape[1]
    ridge[0,0] = 0.
    cov = fit.transpose(0,2,1)@fit
    response = fit.transpose(0,2,1)@y[:,4:25,None]
    coef = np.linalg.solve(cov+ridge,response)
    return (target@coef).squeeze(2)


def observe_frames(values, available, release):
    """Past-125 fitted estimators; each five-point voltage target is held out."""
    # q = release-26; q-124 through q+25, both inclusive.
    idx = release[:,None]+np.arange(-150,0)
    if len(release) and (idx.min()<0 or release.max()>=len(values)):
        raise ValueError('invalid existing feedback clock')
    use = (available[idx] & np.isfinite(values[idx])).all((1,2))
    bins = values[idx[use]].reshape(-1,30,5,len(COLS)).mean(2,dtype=np.float64)
    current, _, voltage, *_ = bins.transpose(2,0,1)
    past_v = voltage[:,:25]
    base_v, scale_v = past_v.mean(1,keepdims=True), np.maximum(past_v.std(1,keepdims=True), .05)
    y = (voltage-base_v)/scale_v
    i = current-current[:,:25].mean(1,keepdims=True)
    di = np.diff(current,axis=1,prepend=current[:,:1])
    scale_i = np.maximum(current[:,:25].std(1,keepdims=True),1e-6*np.abs(current[:,:25]).mean(1,keepdims=True)+1.)
    scale_di = np.maximum(di[:,1:25].std(1,keepdims=True), .05*scale_i)
    rad = bins[:,:,3:]
    dr = np.diff(rad,axis=1,prepend=rad[:,:1])
    dr /= np.maximum(dr[:,1:25].std(1,keepdims=True),
                     .05*np.abs(rad[:,:25]).mean(1,keepdims=True)+1e-12)
    dr = np.clip(dr,-5.,5.)
    yy = y[:,25:]
    previous = np.roll(y,1,axis=1)
    old = np.stack((np.ones_like(y),previous),axis=2)
    circuit = np.concatenate((old,np.stack((i/scale_i,di/scale_di),axis=2)),axis=2)
    by_rad = tuple(np.concatenate((circuit,np.roll(dr,k,axis=1)),axis=2) for k in (1,3))
    estimates = np.stack((previous[:,25:], *(_estimate(x,y,x[:,25:]) for x in (old,circuit,*by_rad))),axis=1)
    losses = np.square(yy[:,None]-estimates).mean(2)
    raw = yy-estimates[:,1]
    residual = yy-estimates[:,2]
    motion = dr[:,25:]
    raw_order, raw_shifted = aligned_scores(raw,motion)
    aligned, shifted = aligned_scores(residual,motion)

    def marginal(x):
        return np.tanh(np.stack((x.mean(1), np.abs(x).max(1), x[:,-1]),axis=1).reshape(len(x),-1)) if len(x) else (
            np.empty((0,3*int(np.prod(x.shape[2:])))))

    amplitudes = np.tanh(np.sqrt(np.c_[np.square(np.diff(y[:,24:],axis=1)).mean(1),
                                     np.square(motion).mean(1)]))
    context = np.c_[amplitudes,marginal(yy),marginal(raw),marginal(residual),marginal(motion)]
    return dict(release=release[use], loss=losses, predictions=estimates, amplitudes=amplitudes,
                features=context,raw_order=raw_order,raw_shifted=raw_shifted,
                conditional_order=aligned,conditional_shifted=shifted,
                current_change=np.abs(current[:,29])-np.abs(current[:,24]))


def observe(out):
    started = time.monotonic()
    lock = lag.load_inputs()
    p,s = (read(REFERENCE/k) for k in ('protocol.json','seal.json'))
    ref = Path(p['reference'])
    anchor = read(ref/'seal.json')
    coords = joint.BRIDGE/'decoders.npz'
    bridge = read(joint.BRIDGE/'seal.json')
    if (p['order']!=lock['ids'] or p['startup']!=20 or len(s['curve_sha256'])!=360
            or p['reference_seal']!=sha(ref/'seal.json') or bridge['assets'][coords.name]!=sha(coords)):
        raise ValueError('locked stream and context required')
    with np.load(coords,allow_pickle=False) as z:
        center,scale = z['center'],z['scale']
    out.mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_CURRENT_CONDITIONED_VOLTAGE_DIAGNOSTIC',
        parent=str(REFERENCE),parent_seal=sha(REFERENCE/'seal.json'),
        inherited_reference=str(ref),inherited_seal=p['reference_seal'],
        order=p['order'],startup=p['startup'],calibration=p['calibration'],original=p['original'],
        parent_arm=PARENT,source_sha256=p['source_sha256'],coordinate_sha256=sha(coords),
        fields=[io23.RAW23[k] for k in COLS],block=BLOCK,calipers=CALIPERS.tolist(),ridge=RIDGE,
        clock='use archived q+26 release; fit preceding 125 points; target is q+1 through q+25; '
              'all five target bins complete before decision; repeat error pairing at nominal +5 ms',
        predictors=PREDICTIONS,probes=MODELS,
        physical_scope='empirical conditional loop voltage, no resistance, control setpoint or radiation power inference',
        access='raw signals only before feature seal; pair models use labels from earlier completed 85-shot blocks'))
    rows, witnesses, releases = [],{},0
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed outcome access')):
        for rank,sid in enumerate(lock['ids'],1):
            if rank<=p['startup']:
                continue
            name = f'{rank:04d}_{sid}.npz'
            path = io23.path_for('EAST',sid)
            a,b = REFERENCE/'shots'/name,ref/'shots'/name
            if (sha(path)!=lock['hashes'][str(sid)] or sha(a)!=s['curve_sha256'][name]
                    or sha(b)!=anchor['curve_sha256'][name]):
                raise ValueError(f'locked data changed: {sid}')
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            mapped = lag.prepare_actual(raw,rank,lock['mapping'])
            _,_,physical,available = shot_coordinates(mapped,center,scale)
            with np.load(a,allow_pickle=False) as z,np.load(b,allow_pickle=False) as w:
                np.testing.assert_array_equal(raw['times'],w['times'])
                release = z['release']
                np.testing.assert_array_equal(release,w['release'])
                releases += len(release)
                curve = z[PARENT]
            x,m = raw['values'][:,COLS],raw['available'][:,COLS]
            got = observe_frames(x,m,release)
            if rank in (21,120,360):
                stop = min(410,len(x))
                short = observe_frames(x[:stop],m[:stop],release[release<stop])
                for k,v in short.items():
                    np.testing.assert_array_equal(v,got[k][got['release']<stop])
                witnesses[str(sid)] = stop
            r = got['release']
            if not available[r].all():
                raise ValueError('current and field context missing at released frame')
            i = physical[r[:,None]+np.arange(-31,1),0].reshape(-1,2,16).mean(2)
            context = np.c_[logit(curve[r].astype(float).clip(1e-6,1.-1e-6)), np.tanh(physical[r]),
                            np.tanh(4*(i[:,1]-i[:,0])), np.log1p(r)]
            rows.append(dict(rank=np.full(len(r),rank),shot=np.full(len(r),sid),time=raw['times'][r],
                marginal=np.c_[context,got['features']],match=np.c_[context,got['amplitudes']],**got))
            if rank%85==0 or rank==360:
                print(dict(stage='voltage_frames',rank=rank),flush=True)
    data = excursion._stack(rows)
    if not all(np.isfinite(v).all() for v in data.values()):
        raise FloatingPointError('nonfinite voltage-response archive')
    joint._npz(out/'frames.npz',**data)
    write(out/'seal.json',dict(prefix_witnesses=witnesses,releases=releases,
        assets={k:sha(out/k) for k in ('frames.npz','protocol.json','executed_probe.py')},
        seconds=time.monotonic()-started))


def reveal(p, rows, lo, hi, known):
    result = {k:np.zeros(len(rows['rank']),bool) for k in ('positive','late','miss','safe','fp','hard5','fp5','late30')}
    records = []
    for rank in range(lo,hi):
        sid = p['order'][rank-1]
        raw,y = Path(p['parent'])/'shots'/f'{rank:04d}_{sid}.npz', rows['rank']==rank
        if sha(raw)!=known[raw.name]:
            raise ValueError(f'parent curve changed: {sid}')
        d,down,t = hdf._raw_truth(sid)
        with np.load(raw,allow_pickle=False) as z:
            curve = z[PARENT]
        threshold = p['original'] if rank<=p['startup'] else p['calibration']['0.2']
        r = rows['release'][y]
        np.testing.assert_array_equal(rows['time'][y],t[r])
        clocks = joint.case_support(t,curve,r,d,down,threshold)
        raw_score = dict(shot=sid,disruptive=d,down_s=down,**score_curve(t,curve,d,down,threshold))
        records.append(raw_score)
        result['positive'][y] = clocks['positive']
        result['safe'][y] = clocks['safe']
        result['late'][y] = clocks['positive'] & clocks['late'] & raw_score['alarm']
        result['miss'][y] = clocks['positive'] & clocks['late'] & (not raw_score['alarm'])
        result['fp'][y] = clocks['safe'] & clocks['fp']
        if d:
            lead = np.round((down-t[r])*1000.,9)
            result['hard5'][y] = clocks['positive'] & clocks['late'] & (lead>35.)
            result['late30'][y] = clocks['late'] & (lead>0.) & (lead<=30.)
        elif clocks['fp']:
            result['fp5'][y] = (t[clocks['first']]-t[r])>=.005
    return {**rows,**result},records


def matched(rows, a, b, *, width=len(CALIPERS)):
    x,y = (np.flatnonzero(v) for v in (a,b))
    idx = excursion.disjoint_pairs(rows['match'][x,:width],rows['match'][y,:width],
                                  rows['shot'][x],rows['shot'][y],CALIPERS[:width])
    return x[idx[:,0]],y[idx[:,1]]


def _design(rows, k):
    return rows['marginal'] if k=='marginal' else np.c_[rows['marginal'],rows[k]]


def paired(rows, scores, a, b, *, width=len(CALIPERS)):
    x,y = matched(rows,a,b,width=width)
    gap = {k:v[x]-v[y] for k,v in scores.items()}
    return dict(pairs=len(x),candidate_positive=len(np.unique(rows['shot'][a])),
        candidate_safe=len(np.unique(rows['shot'][b])),
        ordering={k:dict(correct=int((v>0.).sum()),ties=int((v==0.).sum())) for k,v in gap.items()},
        shot_pairs=np.c_[rows['shot'][x],rows['shot'][y]].tolist(),
        release_pairs=np.c_[rows['release'][x],rows['release'][y]].tolist(),
        margins={k:v.tolist() for k,v in gap.items()})


def error_summary(rows, mask):
    r = rows['rank'][mask]
    loss = np.stack([rows['loss'][mask][r==j].mean(0) for j in np.unique(r)]) if len(r) else (
        np.empty((0,len(PREDICTIONS))))
    if not len(loss):
        return dict(windows=0,shots=0)
    return dict(windows=len(r),shots=len(loss),
        mean_mse=dict(zip(PREDICTIONS,loss.mean(0).tolist(),strict=True)),
        shot_improvements={'current_vs_history':int((loss[:,2]<loss[:,1]).sum()),
                           'lead_vs_current':int((loss[:,3]<loss[:,2]).sum()),
                           'lead_vs_older':int((loss[:,3]<loss[:,4]).sum())})


def score(out):
    p,s = (read(out/k) for k in ('protocol.json','seal.json'))
    if (p['parent_seal']!=sha(REFERENCE/'seal.json') or
            any(sha(out/k)!=v for k,v in s['assets'].items())):
        raise ValueError('sealed input changed')
    old = read(REFERENCE/'seal.json')['curve_sha256']
    with np.load(out/'frames.npz',allow_pickle=False) as f:
        data = {k:f[k] for k in f.files if k!='predictions'}
    records = reveal(p,{k:v[:0] for k,v in data.items()},1,p['startup']+1,old)[1]
    history,blocks = [],[]
    for lo in range(p['startup']+1,len(p['order'])+1,p['block']):
        hi = min(lo+p['block'],len(p['order'])+1)
        active = (data['rank']>=lo)&(data['rank']<hi)
        now = {k:v[active] for k,v in data.items()}
        forecasts = {'parent':now['marginal'][:,0]}
        fit, fits = None,0
        if history:
            train = excursion._stack(history)
            if train['rank'].max()>=lo:
                raise ValueError('future or same-block teaching')
            x,y = matched(train,train['positive'],train['safe'],width=5)
            fits = len(x)
            if len(x):
                fit = dict(n=len(x),history_ranks=np.c_[train['rank'][x],train['rank'][y]].tolist())
                for k in MODELS:
                    w = _design(train,k)
                    coef = excursion.pair_fit(w[x],w[y])
                    forecasts[k] = _design(now,k)@coef
                    fit[k] = coef.tolist()
        f = out/f'block_{lo:03d}_predictions.npz'
        joint._npz(f,rank=now['rank'],release=now['release'],**forecasts)
        revealed, rr = reveal(p,now,lo,hi,old)
        history.append(revealed)
        records.extend(rr)
        errors = revealed['late'] | revealed['miss']
        blocks.append(dict(start=lo,end=hi-1,fit=fit,forecast_sha256=sha(f), forecast=f.name,
            broad=paired(revealed,forecasts,revealed['positive'],revealed['safe'],width=5),
            hard=paired(revealed,forecasts,errors,revealed['fp'],width=5),
            hard_amplitude_matched=paired(revealed,forecasts,errors,revealed['fp']),
            hard_plus5ms=paired(revealed,forecasts,revealed['hard5'],revealed['fp5'],width=5),
            late=paired(revealed,forecasts,revealed['late'],revealed['fp'],width=5),
            miss=paired(revealed,forecasts,revealed['miss'],revealed['fp'],width=5)))
        print(dict(stage='voltage_score',start=lo,prior_pairs=fits,hard=blocks[-1]['hard']['pairs']),flush=True)
    if records!=read(REFERENCE/'per_shot.json')['0.2'][PARENT]:
        raise ValueError('sealed parent first alarms or raw HDF5 outcomes changed')
    r = excursion._stack(history)
    masks = dict(all=np.ones(len(r['rank']),bool),timely=r['positive'],late=r['late'],
                 miss=r['miss'],before_fp=r['fp'],after_deadline=r['late30'],
                 decreasing=r['current_change']<0.,nondecreasing=r['current_change']>=0.)
    write(out/'result.json',dict(status='COMPLETED_CURRENT_CONDITIONED_VOLTAGE_DIAGNOSTIC',
        raw_shots=len(records),release_slots=s['releases'],observed=len(r['rank']),
        parent=dict(timely=sum(x['timely'] for x in records),
                    false_alarm=sum(x['alarm'] and not x['disruptive'] for x in records)),
        conditional_estimation={k:error_summary(r,v) for k,v in masks.items()},blocks=blocks))


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    a = ap.parse_args()
    observe(a.out.resolve())
    score(a.out.resolve())
