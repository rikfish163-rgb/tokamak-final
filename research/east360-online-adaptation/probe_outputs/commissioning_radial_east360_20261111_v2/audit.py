"""Independently verify first alarms and measure conditional direction evidence."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.risk_aligned_online import io23


HERE = Path(__file__).resolve().parent
DEADLINES = (15,20,25,30)
PAIR_CALIPERS = np.array([.5,.5,.125,.5,np.log(2.),np.log(2.)])


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched(data):
    keys = ('radial_fit','two_step_direction','last_radial_change',
            'distance','last_step','prior_step','fixed_risk',
            'signed_latent_effect','unsigned_latent_effect')
    parts = {k:np.concatenate([r[k] for r in data]) for k in data[0]}
    d,s = np.flatnonzero(parts['timely']),np.flatnonzero(parts['safe'])
    x = parts['covariates']
    pairs, distance = [], []
    for i in d:
        z = (x[s]-x[i])/PAIR_CALIPERS
        possible = np.flatnonzero(np.max(abs(z),axis=1)<=1.)
        if len(possible):
            j = possible[np.argmin(np.square(z[possible]).sum(1))]
            pairs.append((i,s[j]))
            distance.append(float(np.linalg.norm(z[j])))
    i,j = np.asarray(pairs,dtype=int).reshape(-1,2).T
    scores, sid = parts['scores'],parts['shot']

    def summary(keep):
        pos,neg = i[keep],j[keep]
        if not len(pos):
            return dict(pairs=0,disruptive_shots=0,safe_shots=0)
        a,b = scores[pos],scores[neg]
        score = (a>b)+.5*(a==b)
        per = np.stack([score[sid[pos]==k].mean(0) for k in np.unique(sid[pos])])
        return dict(pairs=len(pos),disruptive_shots=len(per),safe_shots=len(np.unique(sid[neg])),
                    shot_equal_concordance=dict(zip(keys,per.mean(0).tolist(),strict=True)))

    return dict(calipers=PAIR_CALIPERS.tolist(),safe_reuse=True,available_timely_windows=len(d),
                available_timely_shots=len(np.unique(sid[d])),
                match_distance_median=float(np.median(distance)) if distance else None,
                all=summary(np.ones(len(i),bool)),old_missed_or_late=summary(parts['old_fail'][i]))


def run():
    p,s,per,lead = (read(HERE/k) for k in ('protocol.json','seal.json','per_shot.json','lead_sensitivity.json'))
    old = Path(p['reference'])
    op = read(old/'protocol.json')
    old_seal = read(old/'seal.json')['curve_sha256']
    assert p['order']==op['order'] and p['original']==op['original'] and p['calibration']==op['calibration']
    assert sha(HERE/'mapper.npz')==s['mapper_sha256']
    (readout,) = [Path(k) for k in p['assets'] if Path(k).name=='readout.npz']
    assert sha(readout)==p['assets'][str(readout)]
    with np.load(readout,allow_pickle=False) as f:
        beta_norm = float(np.linalg.norm(f['coef'][-25:-1].astype(np.float64)))
    cutoff = p['calibration']['0.2']
    boundary = np.log(cutoff/(1.-cutoff))
    counts = {b:{k:Counter() for k in p['arms']} for b in p['calibration']}
    n_check = 0
    chunks,misses = [],[]
    for i,sid in enumerate(p['order']):
        name = f'{i+1:04d}_{sid}.npz'
        f = HERE/'shots'/name
        parent = old/'shots'/name
        assert sha(f)==s['curve_sha256'][name] and sha(parent)==old_seal[name]
        with h5py.File(io23.path_for('EAST',sid),'r') as h:
            ip = h['data/ip']
            st = float(np.asarray(ip.attrs['StartTime']).item())
            hz = float(np.asarray(ip.attrs['SampleRate']).item())
            truth = int(np.asarray(h['meta/IsDisrupt'][()]).item())
            assert truth in (0,1)
            d = bool(truth)
            down = float(np.asarray(h['meta/DownTime'][()]).item()) if d else None
            t = st+np.arange(len(ip))/hz
        with np.load(f,allow_pickle=False) as a,np.load(parent,allow_pickle=False) as b:
            np.testing.assert_array_equal(a['times'],t)
            for k in ('fixed','inherited','legacy_current4'):
                np.testing.assert_array_equal(a[k],b[k])
            if i<p['startup']:
                for k in p['arms']:
                    np.testing.assert_array_equal(a[k],a['fixed'])
            for cap,tau in p['calibration'].items():
                threshold = p['original'] if i<p['startup'] else tau
                for k in p['arms']:
                    q = a[k]>=threshold
                    if d:
                        q &= (down-t)*1000. >= -1.e-6
                    ix = np.flatnonzero(q)
                    lead_ms = round(float((down-t[ix[0]])*1000.),9) if d and len(ix) else None
                    row = dict(shot=sid,disruptive=d,down_s=down,
                               alarm=bool(len(ix)),lead_ms=lead_ms,timely=bool(lead_ms is not None and lead_ms>30))
                    assert row == per[cap][k][i], (cap,k,sid)
                    counts[cap][k]['fp']+=bool(not d and len(ix))
                    for deadline in DEADLINES:
                        counts[cap][k][str(deadline)] += bool(lead_ms is not None and lead_ms>deadline)
                    n_check += 1
            if i<p['startup']:
                continue
            release = a['release']
            eligible = a['eligible']
            assert (a['reference_end'][eligible]<=release[eligible]-1-3*p['stride']).all()
            assert (a['step_norm']<=p['cap']+2.e-6).all()
            assert not a['compact'][~eligible].any() and not a['step_norm'][~eligible].any()
            q = release[eligible]
            qlead = (down-t[q])*1000. if d else np.zeros(len(q))
            ctx,amp = a['context'][eligible],a['amplitude'][eligible]
            fixed = a['fixed'][q].astype(np.float64)
            logit = np.log(fixed.clip(1.e-6,1.-1.e-6)/(1.-fixed.clip(1.e-6,1.-1.e-6)))
            cov = np.column_stack((ctx,logit,np.log(amp[:,:2]+1.e-6)))
            original = a['legacy_current4'][q]
            scores = np.column_stack((a['compact'][eligible,:3],amp,fixed,
                                       a['response_plus_old'][q]-original,a['amplitude_plus_old'][q]-original))
            if d and not per['0.2']['legacy_current4'][i]['timely']:
                prob = a['legacy_current4'].astype(np.float64).clip(1.e-7,1.-1.e-7)
                margin = boundary-(np.log(prob)-np.log1p(-prob))
                step = np.zeros(len(t))
                for end,radius in zip(release,a['step_norm'],strict=True):
                    step[end:end+p['stride']] = radius
                advance = (down-t)*1000.
                timely = advance>30.+1.e-6
                near = timely & (advance<=125.+1.e-6)
                possible = margin<=beta_norm*step+1.e-6
                misses.append(dict(shot=sid,median_near_shift=float(np.median(step[near])),
                    near_has_action=bool(np.any(step[near]>0.)),near_bound_possible=bool(np.any(near&possible)),
                    timely_bound_possible=bool(np.any(timely&possible)),
                    new_timely=per['0.2']['response_plus_old'][i]['timely']))
            chunks.append(dict(shot=np.full(len(q),sid),covariates=cov,scores=scores,
                timely=(qlead>30.+1.e-6)&(qlead<=125.+1.e-6),safe=np.full(len(q),not d),
                old_fail=np.full(len(q),d and not per['0.2']['legacy_current4'][i]['timely'])))
    for cap,g in counts.items():
        for k,r in g.items():
            z = lead['metrics'][cap][k]
            assert z['fp']==r['fp'] and all(z['by_deadline'][str(a)]['tp']==r[str(a)] for a in DEADLINES)
    ability = dict(
        scope='upper bound on fixed-dose hidden move, not a learned oracle',
        readout_norm=beta_norm, old_missed_or_late=len(misses),
        near_has_action=sum(r['near_has_action'] for r in misses),
        near_bound_possible=sum(r['near_bound_possible'] for r in misses),
        timely_bound_possible=sum(r['timely_bound_possible'] for r in misses),
        actual_rescues=sum(r['new_timely'] for r in misses),per_shot=misses)
    out = dict(status='HDF5_FIRST_ALARMS_AND_CONDITIONAL_DIRECTION_CHECKED',scored_curves=n_check,
               counts=counts,pair_diagnostic=matched(chunks),dose_reachability=ability)
    with (HERE/'independent_audit.json').open('x') as f:
        json.dump(out,f,indent=2)
    print(dict(scored_curves=n_check,counts_main=counts['0.2'],pair_diagnostic=out['pair_diagnostic']))


if __name__ == '__main__':
    run()
