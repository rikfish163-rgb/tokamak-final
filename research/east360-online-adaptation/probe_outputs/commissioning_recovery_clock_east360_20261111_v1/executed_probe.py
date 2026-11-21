"""Does the observed lifetime of a deviation help distinguish actual warning errors?"""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np

from . import commissioning_arrival_paths_20261111 as arrival
from . import commissioning_coincident_innovation_20261111 as joint
from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_excursion_order_20261111 as excursion
from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .commissioning_transition_source_20261110 import shot_coordinates
from .history_gradient_metric_20261102.run import read, sha, write


MEAN_POINTS, REFERENCE_POINTS, REFERENCE_LAG, CONFIRM = 5, 125, 10, 3
HIGH, LOW, MEMORY = 3., 1.5, 125
END_OF_RECORD, RETURNED, OBSERVATION_GAP = range(3)
TRACE = ('state', 'distance', 'age', 'return_age', 'return_duration')


def rolling_reference(z, observed):
    """Five-point measurement and earlier, disjoint 125-point level reference."""
    observed = observed & np.isfinite(z)
    v = np.where(observed,z,0.)
    total, square, count = (np.vstack((np.zeros((1,z.shape[1])),np.cumsum(a,axis=0,dtype=float)))
                            for a in (v,np.square(v),observed))
    stop = np.arange(1,len(z)+1)
    start = (stop-MEAN_POINTS).clip(min=0)
    value = (total[stop]-total[start])/MEAN_POINTS
    observed_now = (count[stop]-count[start])==MEAN_POINTS
    hi = (stop-REFERENCE_LAG).clip(min=0)
    lo = (hi-REFERENCE_POINTS).clip(min=0)
    center = (total[hi]-total[lo])/REFERENCE_POINTS
    variance = (square[hi]-square[lo])/REFERENCE_POINTS-np.square(center)
    spread = np.sqrt(np.maximum(variance,excursion.FLOOR**2))
    return dict(value=value,observed_now=observed_now,center=center,spread=spread,
                reference_known=(count[hi]-count[lo])==REFERENCE_POINTS)


def _distance(delta):
    return float(np.sqrt((delta@delta)/len(delta)))


def episode_trace(value, observed_now, center, spread, reference_known):
    """Confirm against a frozen pre-excursion range; a gap is not a return."""
    n = len(value)
    state = np.full(n,-1,dtype=np.int8)
    distance = np.zeros(n)
    age, return_age, return_duration = (np.full(n,-1,dtype=np.int32) for _ in range(3))
    episodes = []
    active = candidate = anchor = last_return = last_duration = -1
    below = 0
    for t in range(n):
        if not observed_now[t]:
            if active>=0:
                episodes.append((active,t-1,OBSERVATION_GAP))
            active = candidate = last_return = last_duration = -1
            below = 0
            continue
        if active>=0 or candidate>=0:
            dist = _distance((value[t]-center[anchor])/spread[anchor])
            state[t],distance[t] = 0,dist
            if active>=0:
                below = below+1 if dist<=LOW else 0
                if below==CONFIRM:
                    last_return,last_duration = t,t-active
                    episodes.append((active,t,RETURNED))
                    active,below = -1,0
            elif dist<=HIGH:
                candidate = -1
            elif t-candidate+1==CONFIRM:
                active,candidate = t,-1
        elif reference_known[t]:
            dist = _distance((value[t]-center[t])/spread[t])
            state[t],distance[t] = 0,dist
            if dist>HIGH:
                anchor = candidate = t
        if active>=0:
            state[t],age[t] = 1,t-active
        if state[t]>=0 and last_return>=0 and t-last_return<MEMORY:
            return_age[t],return_duration[t] = t-last_return,last_duration
    if active>=0:
        # Record-end censoring is for the retrospective event inventory only.
        episodes.append((active,n-1,END_OF_RECORD))
    return dict(state=state,distance=distance,age=age,return_age=return_age,
                return_duration=return_duration,episodes=np.asarray(episodes,dtype=np.int64).reshape(-1,3))


def sample_at_release(trace, releases):
    """A newly confirmed event at t is eligible from t+1, not earlier."""
    return {k:trace[k][releases-1] for k in TRACE}


def _duration(x):
    return np.log1p(np.clip(x+1,0,MEMORY))/np.log1p(MEMORY)


def shot_views(z, mask, releases):
    inputs = rolling_reference(z,mask)
    instant, window, state, duration, inventory, flags = [],[],[],[],[],[]
    old = releases[:,None]+np.arange(-lag.H,0)
    for g in range(len(excursion.FAMILIES)):
        use = excursion.GROUPS==g
        v, center, spread = (inputs[k][:,use] for k in ('value','center','spread'))
        o,c = (inputs[k][:,use].all(1) for k in ('observed_now','reference_known'))
        known = o&c
        d = np.where(known,np.log1p(np.sqrt(np.square((v-center)/spread).mean(1))),0.)
        q = releases-1
        instant.append(np.c_[known[q],d[q]])
        recent, full = d[old],known[old].all(1)
        window.append(np.c_[full,np.where(full[:,None],
            np.c_[recent.mean(1),recent.max(1),recent[:,-1]-recent[:,0]],0.)])
        trace = episode_trace(v,o,center,spread,c)
        snap = sample_at_release(trace,releases)
        active, returned = snap['state']==1,snap['return_age']>=0
        state.append(np.c_[snap['state']>=0,active,np.log1p(snap['distance'])])
        duration.append(np.c_[_duration(snap['age']),returned,
            np.where(returned,1.-snap['return_age']/MEMORY,0.),_duration(snap['return_duration'])])
        flags.append(np.c_[snap['state']>=0,active,returned])
        e = trace['episodes']
        inventory.append(np.c_[np.full(len(e),g),e])
    return dict(instant=np.column_stack(instant),window=np.column_stack(window),
                state=np.column_stack(state),duration=np.column_stack(duration),
                flags=np.stack(flags,axis=1),episodes=np.concatenate(inventory))


def observe(prior, out):
    started = time.monotonic()
    p,frames = arrival._prior(prior)
    lock = lag.load_inputs()
    parent = Path(p['reference'])
    coord = joint.BRIDGE/'decoders.npz'
    if (p['order']!=lock['ids'] or sha(parent/'seal.json')!=p['reference_seal']
            or sha(coord)!=p['coordinate_sha256']):
        raise ValueError('cohort, fixed score or normalization changed')
    with np.load(coord,allow_pickle=False) as f:
        center,scale = f['center'],f['scale']
    out.mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_DEVIATION_LIFETIME_SHADOW_NOT_ALARMS',
        prior=str(prior),prior_seal=sha(prior/'observation_seal.json'),order=p['order'],startup=p['startup'],
        producer_sha256=sha(Path(__file__)),release=p['release'],stride=p['stride'],
        main_threshold=p['main_threshold'],original=p['original'],parent=str(parent),
        parent_seal=p['reference_seal'],families=excursion.FAMILIES,block_shots=joint.BLOCK_SHOTS,
        windows=dict(level=MEAN_POINTS,reference=REFERENCE_POINTS,reference_lag=REFERENCE_LAG,
                     short=lag.H,return_memory=MEMORY,confirm=CONFIRM),
        standardized_rms_thresholds=dict(start_above=HIGH,return_at_most=LOW,sd_floor=excursion.FLOOR),
        events=dict(record_end=END_OF_RECORD,confirmed_return=RETURNED,observation_gap=OBSERVATION_GAP),
        fit='same earlier broad disjoint shot pairs; postseal near-deadline late/no-warning vs true FP',
        clock='signals up to release-1; start and return confirmed now, never backdated; '
              'record-end and gap are not returns; current block outcomes absent from fit',
        arms=['amplitude','fixed_window','episode_state','lifetime']))
    rows,episodes,witnesses = [],[],{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed outcome access')):
        for rank,sid in enumerate(lock['ids'],1):
            if rank<=p['startup']:
                continue
            raw_path = io23.path_for('EAST',sid)
            if sha(raw_path)!=lock['hashes'][str(sid)]:
                raise ValueError('sealed signal changed')
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            mapped = lag.prepare_actual(raw,rank,lock['mapping'])
            z,m,_,_ = shot_coordinates(mapped,center,scale)
            idx = frames['rank']==rank
            q = frames['release'][idx]
            np.testing.assert_array_equal(raw['times'][q],frames['time'][idx])
            result = shot_views(z,m,q)
            e = result.pop('episodes')
            episodes.append(np.c_[np.full(len(e),rank),e])
            if rank in (21,120,360):
                end = min(len(z),410)
                short = shot_views(z[:end],m[:end],q[q<end])
                for k in result:
                    np.testing.assert_array_equal(result[k][q<end],short[k])
                witnesses[str(sid)] = end
            rows.append(result)
            if rank%60==0:
                print(dict(stage='seal_lifetimes',rank=rank,seconds=round(time.monotonic()-started,2)),
                      flush=True)
    joint._npz(out/'views.npz',rank=frames['rank'],release=frames['release'],
               episodes=np.concatenate(episodes),**excursion._stack(rows))
    write(out/'seal.json',dict(prefix_witnesses=witnesses,seconds=time.monotonic()-started,
        assets={k:sha(out/k) for k in ('views.npz','protocol.json','executed_probe.py')}))


def _designs(frames, views):
    a = np.c_[frames['marginal'],np.log1p(frames['release']),views['instant']]
    b = np.c_[a,views['window']]
    c = np.c_[b,views['state']]
    return dict(amplitude=a,fixed_window=b,episode_state=c,lifetime=np.c_[c,views['duration']])


def score(out):
    p,s = [read(out/k) for k in ('protocol.json','seal.json')]
    if (any(sha(out/k)!=v for k,v in s['assets'].items())
            or sha(out/'executed_probe.py')!=p['producer_sha256']):
        raise ValueError('sealed observations changed')
    prior = Path(p['prior'])
    if sha(prior/'observation_seal.json')!=p['prior_seal']:
        raise ValueError('matched observations changed')
    _,frames = arrival._prior(prior)
    with np.load(out/'views.npz',allow_pickle=False) as f:
        for k in ('rank','release'):
            np.testing.assert_array_equal(f[k],frames[k])
        views = dict(f)
    parent = Path(p['parent'])
    if sha(parent/'seal.json')!=p['parent_seal']:
        raise ValueError('parent timing seal changed')
    parent_seal = read(parent/'seal.json')['curve_sha256']
    rank_of_frame = frames['rank']
    fitted = _designs(frames,views)
    known = dict(positive=np.zeros(len(rank_of_frame),bool),safe=np.zeros(len(rank_of_frame),bool),
                 late=np.zeros(len(rank_of_frame),bool),miss=np.zeros(len(rank_of_frame),bool),
                 fp=np.zeros(len(rank_of_frame),bool))
    block_size = p['block_shots']
    starts = [1,*range(p['startup']+1,len(p['order'])+1,block_size)]
    records,blocks = [],[]
    for start,stop in zip(starts,starts[1:]+[len(p['order'])+1],strict=True):
        past = rank_of_frame<start
        cur = (rank_of_frame>=start)&(rank_of_frame<stop)
        ctx = {k:v[cur] for k,v in frames.items()}
        prev = {k:v[past] for k,v in frames.items()}
        a,b = joint._pair(prev,known['positive'][past],known['safe'][past])
        logits,coefs = dict(fixed=ctx['marginal'][:,0]),{}
        if len(a):
            for k,x in fitted.items():
                coef = excursion.pair_fit(x[past][a],x[past][b])
                coefs[k] = coef.tolist()
                logits[k] = x[cur]@coef
        file = out/f'block_{start:03d}_predictions.npz'
        joint._npz(file,rank=ctx['rank'],release=ctx['release'],**logits)
        digest = sha(file)
        for rank in range(start,stop):
            sid = p['order'][rank-1]
            name = f'{rank:04d}_{sid}.npz'
            curve = parent/'shots'/name
            if sha(curve)!=parent_seal[name]:
                raise ValueError('changed frozen probability trace')
            d,down,t = hdf._raw_truth(sid)
            at = rank_of_frame==rank
            q = frames['release'][at]
            threshold = p['original'] if rank<=p['startup'] else p['main_threshold']
            with np.load(curve,allow_pickle=False) as f:
                np.testing.assert_array_equal(t,f['times'])
                r = joint.case_support(t,f['fixed'],q,d,down,threshold)
            known['positive'][at],known['safe'][at] = r['positive'],r['safe']
            known['late'][at] = r['positive'] & r['late'] & (r['first'] is not None)
            known['miss'][at] = r['positive'] & (r['first'] is None)
            known['fp'][at] = r['safe'] & r['fp']
            supported = r['positive'] if d else r['safe']
            flags = views['flags'][at][supported]
            records.append(dict(rank=rank,shot=sid,disruptive=d,first=r['first'],
                first_lead_ms=r['first_lead_ms'],late=r['late'],fp=r['fp'],
                pre_feedback_fp=r['pre_feedback_fp'],any_timely_support=bool(r['positive'].any()),
                eligible=len(flags),episode_known=flags[:,:,0].any(0).tolist(),
                active=flags[:,:,1].any(0).tolist(),recent_return=flags[:,:,2].any(0).tolist()))
        if cur.any():
            masks = {k:v[cur] for k,v in known.items()}
            tests = {k:joint._evaluate(ctx,masks[i],masks[j],logits) for k,i,j in
                (('late_vs_fp','late','fp'),('no_alarm_vs_fp','miss','fp'),('all_vs_safe','positive','safe'))}
            blocks.append(dict(first_rank=start,last_rank=stop-1,fit_pairs=len(a),
                fit_shots=np.c_[prev['shot'][a],prev['shot'][b]].tolist(),
                coefficients=coefs,forecasts=file.name,sha256=digest,tests=tests))
            print(dict(stage='compare_after_seal',start=start,fit_pairs=len(a),
                       late_vs_fp=tests['late_vs_fp']['pairs']),flush=True)
    inherited = read(prior/'result.json')['per_shot_clock']
    if ([{k:r[k] for k in old} for r,old in zip(records,inherited,strict=True)]!=inherited
            or sum(r['disruptive'] for r in records)!=180 or len(records)!=360):
        raise ValueError('independent HDF5 outcomes or first warnings differ from the fixed 360')
    event = views['episodes']
    totals = {g:np.bincount(event[event[:,1]==i,-1],minlength=3).tolist()
              for i,g in enumerate(excursion.FAMILIES)}
    write(out/'result.json',dict(status='COMPLETED_EAST360_LIFETIME_SHADOW_NOT_ALARMS',
        frames=len(rank_of_frame),verified_raw_shots=len(records),episode_totals=totals,
        baseline=dict(timely=sum(r['disruptive'] and not r['late'] for r in records),
                      fp=sum(r['fp'] for r in records)),blocks=blocks,per_shot=records))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('observe','score'))
    parser.add_argument('out',type=Path)
    parser.add_argument('--prior',type=Path,default=arrival.REFERENCE)
    a = parser.parse_args()
    observe(a.prior.resolve(),a.out) if a.stage=='observe' else score(a.out)
