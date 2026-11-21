"""Audit when a directional diagnostic path is observable relative to an irreversible first alarm."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np

from . import commissioning_coincident_innovation_20261111 as joint
from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_excursion_order_20261111 as excursion
from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .commissioning_transition_source_20261110 import shot_coordinates
from .history_gradient_metric_20261102.run import read, sha, write


REFERENCE = io23.ROOT/'probe_outputs/commissioning_coincident_innovation_east360_20261111_v1'
BIN, REFERENCE_POINTS, CONFIRM = 5, 125, 3
MAX_GAP = HOLD = joint.STRIDE
GROUPS = excursion.FAMILIES
ROUTES = ('magnetic_then_position','position_then_magnetic','magnetic_position_same_tick',
          'radiation_then_voltage','voltage_then_radiation','radiation_voltage_same_tick')


def confirmed_edges(high):
    """Confirm a sustained excursion now, never at the start of its run."""
    confirmed = np.zeros_like(high,dtype=bool)
    if len(high)>=CONFIRM:
        confirmed[CONFIRM-1:] = np.lib.stride_tricks.sliding_window_view(high,CONFIRM,axis=0).all(-1)
    return confirmed & ~np.concatenate((np.zeros_like(confirmed[:1]),confirmed[:-1]),axis=0)


def arrivals(z, observed):
    """Past-only robust changes in the fixed source coordinate; not physical event labels."""
    visible = observed & np.isfinite(z)
    v = np.where(visible,z,0.)
    value = np.vstack((np.zeros((1,z.shape[1])),np.cumsum(v,axis=0,dtype=float)))
    count = np.vstack((np.zeros((1,z.shape[1]),dtype=int),np.cumsum(visible,axis=0)))
    k = np.arange(2*BIN-1,len(z))
    change = (value[k+1]-2.*value[k-BIN+1]+value[k-2*BIN+1])/BIN
    valid = (count[k+1]-count[k-2*BIN+1])==2*BIN
    strength = np.zeros((len(z),len(GROUPS)))
    available = np.zeros_like(strength,dtype=bool)
    for g in range(len(GROUPS)):
        c = excursion.GROUPS==g
        strength[k,g] = np.sqrt(np.square(change[:,c]).mean(1))
        available[k,g] = valid[:,c].all(1)
    # The baseline uses older differences ending before either current 5-point mean.
    t = np.arange((REFERENCE_POINTS-1)+(2*BIN-1)+2*BIN,len(z))
    high = np.zeros_like(available)
    if len(t):
        start = t-(REFERENCE_POINTS-1)-2*BIN
        history = np.lib.stride_tricks.sliding_window_view(strength,REFERENCE_POINTS,axis=0)[start]
        support = np.lib.stride_tricks.sliding_window_view(available,REFERENCE_POINTS,axis=0)[start]
        center = np.median(history,axis=2)
        mad = np.median(np.abs(history-center[:,:,None]),axis=2)
        threshold = np.maximum(excursion.FLOOR,center+3.*mad)
        high[t] = (strength[t]>threshold) & support.all(2) & available[t]
    return confirmed_edges(high)


def _follows(first, second):
    if not len(first):
        return second[:0]
    j = np.searchsorted(first,second,side='left')-1
    gap = second-first[j.clip(min=0)]
    return second[(j>=0) & (gap>0) & (gap<=MAX_GAP)]


def path_finishes(onsets):
    mag,pos,rad,volt = onsets
    return dict(zip(ROUTES,(_follows(mag,pos),_follows(pos,mag),np.intersect1d(mag,pos),
                            _follows(rad,volt),_follows(volt,rad),np.intersect1d(rad,volt)),strict=True))


def active_at(finished, endpoints):
    """A path can first be used on the point *after* the final confirmation."""
    if not len(finished):
        return np.zeros(len(endpoints),dtype=bool)
    j = np.searchsorted(finished,endpoints,side='left')-1
    age = endpoints-1-finished[j.clip(min=0)]
    return (j>=0) & (age>=0) & (age<HOLD)


def _states(onsets, endpoints):
    sequences = path_finishes(onsets)
    return np.column_stack([active_at(sequences[k],endpoints) for k in ROUTES])


def _prior(path):
    p,seal,checked = [read(path/k) for k in ('protocol.json','observation_seal.json','hdf5_and_support_audit.json')]
    frames = path/'frames.npz'
    if (checked['status']!='PASS' or checked['observation_sha256']!=seal['assets'][frames.name]
            or sha(frames)!=checked['observation_sha256'] or sha(path/'protocol.json')!=seal['assets']['protocol.json']):
        raise ValueError('the verified baseline observation evidence changed')
    with np.load(frames,allow_pickle=False) as f:
        return p,dict(f)


def observe(prior, out):
    started = time.monotonic()
    p,frames = _prior(prior)
    lock = lag.load_inputs()
    ref = Path(p['reference'])
    coord = joint.BRIDGE/'decoders.npz'
    if (p['order']!=lock['ids'] or sha(ref/'seal.json')!=p['reference_seal']
            or sha(coord)!=p['coordinate_sha256']):
        raise ValueError('the locked raw cohort or normalization changed')
    with np.load(coord,allow_pickle=False) as f:
        center, scale = f['center'],f['scale']
    refseal = read(ref/'seal.json')['curve_sha256']
    out.mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_ARRIVAL_PATH_TIMING_NOT_ALARMS',prior=str(prior),
        prior_seal=sha(prior/'observation_seal.json'), order=p['order'], startup=p['startup'],
        fixed_reference=str(ref), fixed_reference_seal=p['reference_seal'],
        fit='past blocks only, broad disjoint shot matching; original score and activity context identical',
        first_effect=joint.FIRST, predecessor_max_points=MAX_GAP, hold_points=HOLD,
        history_points=REFERENCE_POINTS, bin_points=BIN, confirm_points=CONFIRM,
        signal_threshold='past median + 3 MAD of 5-point mean differences; lower bound .05',
        families=GROUPS,routes=ROUTES,
        clock='no backdating; each completed path first acts next sample; same-start event clock '
              'compared to inherited 153+32k observation clock; signal read only before seal',
        producer_sha256=sha(Path(__file__))))
    marks, views, witnesses = [], [], {}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome is not observable')):
        for rank,sid in enumerate(lock['ids'],1):
            name = f'{rank:04d}_{sid}.npz'
            path, old = io23.path_for('EAST',sid),ref/'shots'/name
            if sha(path)!=lock['hashes'][str(sid)] or sha(old)!=refseal[name]:
                raise ValueError('raw diagnostic or fixed reference changed')
            if rank<=p['startup']:
                continue
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            mapped = lag.prepare_actual(raw,rank,lock['mapping'])
            z,m,_,_ = shot_coordinates(mapped,center,scale)
            q = frames['release'][frames['rank']==rank]
            with np.load(old,allow_pickle=False) as f:
                np.testing.assert_array_equal(raw['times'],f['times'])
                np.testing.assert_array_equal(q,f['release'])
            event = arrivals(z,m)
            sets = [np.flatnonzero(x) for x in event.T]
            t,g = np.nonzero(event)
            marks.append(np.c_[np.full(len(t),rank),t,g])
            views.append(_states(sets,q))
            if rank in (21,120,360):
                end = min(len(z),380)
                prefix = arrivals(z[:end],m[:end])
                np.testing.assert_array_equal(prefix,event[:end])
                short_sets = [np.flatnonzero(x) for x in prefix.T]
                np.testing.assert_array_equal(_states(short_sets,q[q<end]),views[-1][q<end])
                witnesses[str(sid)] = end
            if rank%85==0 or rank==360:
                print(dict(stage='seal_causal_events',rank=rank,seconds=round(time.monotonic()-started,2)),flush=True)
    joint._npz(out/'events.npz',events=np.concatenate(marks),rank=frames['rank'],
               release=frames['release'],states=np.concatenate(views))
    write(out/'seal.json',dict(assets={k:sha(out/k) for k in ('protocol.json','executed_probe.py','events.npz')},
        prefix_witnesses=witnesses,seconds=time.monotonic()-started))


def _describe_routes(times, q, onsets, group, down, first):
    paths = path_finishes(onsets)
    live = np.arange(joint.FIRST,len(times))
    live_state = np.column_stack([active_at(v,live) for v in paths.values()])
    step_state = np.column_stack([active_at(v,q) for v in paths.values()])
    if group=='false_alarm':
        return dict(event_at_first=live_state[live==first].any(0).tolist(),
                    clock_at_last_pre_alarm=step_state[q<=first][-1:].any(0).tolist(),
                    confirmation_lead_ms={k:float((times[first]-times[v[-1]])*1000.) if len(v) else None
                                          for k,r in paths.items() for v in (r[r<first],)})
    if down is None:
        return {}
    leads = np.round((down-times)*1000.,9)
    early = (leads>30.) & (leads<=125.)
    late = (leads>=0.) & (leads<=30.)
    if first is not None:
        late &= np.arange(len(times))<=first
    fam = [dict(confirmed_in_early=bool(early[v].any()),confirmed_late=bool(late[v].any()),
                latest_early_lead_ms=float(leads[v[early[v]][-1]]) if early[v].any() else None)
           for v in onsets]
    return dict(families=dict(zip(GROUPS,fam,strict=True)),
                live_early=live_state[early[live]].any(0).tolist(),clock_early=step_state[early[q]].any(0).tolist(),
                clock_early_extra_5ms=step_state[early[q]&(leads[q]>35.)].any(0).tolist(),
                live_late_before_first=live_state[late[live]].any(0).tolist())


def _scores(frames, states, masks):
    marginal = frames['marginal']
    common = np.column_stack((states[:,:3].any(1),states[:,3:].any(1)))
    directed = states[:,[0,3]].astype(float)-states[:,[1,4]].astype(float)
    x = dict(marginal=marginal,unordered=np.c_[marginal,common],ordered=np.c_[marginal,common,directed])
    reversal = np.c_[marginal,common,-directed]
    fixed = dict(fixed=marginal[:,0],forward=states[:,[0,3]].sum(1),reverse=states[:,[1,4]].sum(1),
                 unordered=common.sum(1),**dict(zip(ROUTES,states.astype(float).T,strict=True)))
    runs = []
    for start in range(21,361,joint.BLOCK_SHOTS):
        rank = frames['rank']
        past = rank<start
        train = {k:v[past] for k,v in frames.items()}
        a,b = joint._pair(train,masks['positive'][past],masks['safe'][past])
        take = (rank>=start)&(rank<start+joint.BLOCK_SHOTS)
        sub = {k:v[take] for k,v in frames.items()}
        fits = {}
        if len(a):
            for k,v in x.items():
                coef = excursion.pair_fit(v[past][a],v[past][b])
                fits[k] = v[take]@coef
                if k=='ordered':
                    fits['reversed_test_order'] = reversal[take]@coef
        fit_and_fixed = {**{f'raw_{k}':v[take] for k,v in fixed.items()},**fits}
        runs.append(dict(first_rank=start,fit_pairs=len(a),
            late_vs_false=joint._evaluate(sub,masks['late'][take],masks['false'][take],fit_and_fixed),
            all_vs_safe=joint._evaluate(sub,masks['positive'][take],masks['safe'][take],fit_and_fixed)))
    pairs = {k:joint._evaluate(frames,masks[k],masks['false'],fixed) for k in ('late','no_alarm')}
    return dict(past_only_fits=runs,postseal_fixed_direction=pairs)


def score(out):
    p,seal = [read(out/k) for k in ('protocol.json','seal.json')]
    if any(sha(out/k)!=v for k,v in seal['assets'].items()) or sha(out/'executed_probe.py')!=p['producer_sha256']:
        raise ValueError('the sealed path observations changed')
    prior = Path(p['prior'])
    if sha(prior/'observation_seal.json')!=p['prior_seal']:
        raise ValueError('the paired context has changed')
    p0, frames = _prior(prior)
    with np.load(out/'events.npz',allow_pickle=False) as f:
        marks,states = f['events'],f['states']
        for k in ('rank','release'):
            np.testing.assert_array_equal(f[k],frames[k])
    parent = Path(p['fixed_reference'])
    if sha(parent/'seal.json')!=p['fixed_reference_seal']:
        raise ValueError('the parent timing archive has changed')
    parent_seal = read(parent/'seal.json')['curve_sha256']
    previous = read(prior/'result.json')['per_shot_clock']
    masks = {k:np.zeros(len(states),bool) for k in ('positive','safe','late','no_alarm','false')}
    per_shot, cases = [], []
    for rank,sid in enumerate(p['order'],1):
        d,down,t = hdf._raw_truth(sid)
        name = f'{rank:04d}_{sid}.npz'
        path = parent/'shots'/name
        if sha(path)!=parent_seal[name]:
            raise ValueError('the inherited curve changed')
        at = np.flatnonzero(frames['rank']==rank)
        q = frames['release'][at]
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(t,f['times'])
            static = joint.case_support(t,f['fixed'],q,d,down,
                p0['original'] if rank<=p0['startup'] else p0['main_threshold'])
        r = previous[rank-1]
        if (r['shot']!=sid or r['disruptive']!=d or r['first']!=static['first']
                or r['first_lead_ms']!=static['first_lead_ms']):
            raise ValueError('the raw HDF5 does not reproduce inherited outcome and first warning')
        cases.append(d)
        if rank<=p['startup']:
            continue
        group = ('no_alarm' if d and static['first'] is None else 'late' if static['late'] else
                 'timely' if d else 'false_alarm' if static['fp'] else 'no_false_alarm')
        masks['positive'][at], masks['safe'][at] = static['positive'],static['safe']
        masks['late'][at] = static['positive'] & (group=='late')
        masks['no_alarm'][at] = static['positive'] & (group=='no_alarm')
        masks['false'][at] = static['safe'] & static['fp']
        mark = marks[marks[:,0]==rank]
        onsets = [mark[mark[:,2]==g,1] for g in range(len(GROUPS))]
        np.testing.assert_array_equal(_states(onsets,q),states[at])
        per_shot.append(dict(shot=sid,rank=rank,group=group,first=static['first'],
            onset_counts=[len(v) for v in onsets],**_describe_routes(t,q,onsets,group,down,static['first'])))
    if len(cases)!=360 or sum(cases)!=180:
        raise ValueError('unexpected HDF5 roster')
    summaries = {}
    for group in ('timely','late','no_alarm','false_alarm'):
        selected = [r for r in per_shot if r['group']==group]
        keys = ('live_early','clock_early','clock_early_extra_5ms','live_late_before_first') if group!='false_alarm' else (
            'event_at_first','clock_at_last_pre_alarm')
        summary = dict(shots=len(selected))
        for key in keys:
            a = np.asarray([r[key] for r in selected],bool)
            summary[key] = dict(any=int(a.any(1).sum()),directional=int(a[:,[0,1,3,4]].any(1).sum()),
                                by_route=dict(zip(ROUTES,a.sum(0).tolist(),strict=True)))
        if group!='false_alarm':
            summary['only_late_directional'] = sum(not any(r['live_early'][i] for i in (0,1,3,4))
                and any(r['live_late_before_first'][i] for i in (0,1,3,4)) for r in selected)
        summaries[group] = summary
    comparisons = _scores(frames,states,masks)
    write(out/'result.json',dict(status='COMPLETED_ARRIVAL_PATH_TIMING_NOT_ALARMS',
        verified_raw_shots=len(cases),confirmed_events=len(marks),verified_clock_rows=len(states),
        timing=summaries,comparisons=comparisons,per_shot=per_shot))
    print({k:dict(shots=v['shots'],counts={a:b['directional'] for a,b in v.items() if isinstance(b,dict)})
           for k,v in summaries.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('observe','score'))
    parser.add_argument('out',type=Path)
    parser.add_argument('--prior',type=Path,default=REFERENCE)
    a = parser.parse_args()
    observe(a.prior.resolve(),a.out) if a.stage=='observe' else score(a.out)
