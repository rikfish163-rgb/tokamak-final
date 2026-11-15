"""Do simultaneous diagnostic changes help distinguish the frozen parent's actual errors?"""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit

from . import commissioning_excursion_order_20261111 as excursion
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_response_bridge_20261111 as bridge
from . import io23, simple_data
from .commissioning_transition_data_20261110 import _window_mean
from .commissioning_transition_source_20261110 import shot_coordinates
from .history_gradient_metric_20261102.run import read, sha, write


BRIDGE = io23.ROOT/'probe_outputs/commissioning_response_bridge_east360_20261111_v1'
H, FIRST, STRIDE = lag.H, lag.FIRST_EFFECT, lag.STRIDE
BLOCK_SHOTS = 85
FAMILIES = excursion.FAMILIES
P, Q = np.triu_indices(len(FAMILIES), k=1)
PAIR_NAMES = tuple(f'{FAMILIES[i]}__{FAMILIES[j]}' for i, j in zip(P,Q,strict=True))
CALIPER = np.r_[excursion.CALIPERS[:4], np.full(len(FAMILIES),.20)]
T = np.arange(5,dtype=float)-2.
MODELS = ('marginal','aligned','shifted')


def rotate_families(activity):
    """A deterministic negative control preserving each envelope's multiset and cyclic order."""
    return np.stack([np.roll(activity[:,:,g],g,axis=1) for g in range(len(FAMILIES))],2)


def signatures(activity):
    mean = activity.mean(1)
    centered = activity-mean[:,None]
    slope = (centered*T[None,:,None]).sum(1)/np.square(T).sum()
    remainder = centered-slope[:,None]*T[None,:,None]
    z = remainder/np.maximum(np.linalg.norm(remainder,axis=1,keepdims=True),1e-8)

    def pairs(a):
        return (a[:,:,P]*a[:,:,Q]).sum(1)

    single = np.tanh(np.c_[mean,activity.max(1),slope])
    return single,pairs(z),pairs(rotate_families(z))


def observed_features(z,m,phys,pm):
    release = np.arange(FIRST,len(z),STRIDE,dtype=np.int64)
    q = release-H-1
    hist = q[:,None]+np.arange(-124,1)
    arrive = q[:,None]+np.arange(1,H+1)
    valid = m[hist].all((1,2)) & m[arrive].all((1,2))
    recent,n = _window_mean(phys,pm,release,-15,16)
    before,b = _window_mean(phys,pm,release,-31,16)
    valid &= (n>=8).all(1) & (b[:,0]>=8)
    older = z[hist[valid]].reshape(-1,25,5,z.shape[1]).mean(2)
    newer = z[arrive[valid]].reshape(-1,5,5,z.shape[1]).mean(2)
    old_delta = np.diff(older,axis=1)
    new_delta = np.diff(np.concatenate((older[:,-1:],newer),1),axis=1)
    activity = []
    for g in range(len(FAMILIES)):
        channels = excursion.GROUPS == g
        prev = np.sqrt(np.square(old_delta[:,:,channels]).mean(2))
        now = np.sqrt(np.square(new_delta[:,:,channels]).mean(2))
        scale = np.maximum(np.median(prev,axis=1),excursion.FLOOR)
        activity.append(np.log1p(now/scale[:,None]))
    single, real, shift = signatures(np.stack(activity,2))
    context = np.tanh(np.c_[recent,4.*(recent[:,0]-before[:,0])])[valid]
    return dict(release=release[valid],context=context,single=single,aligned=real,shifted=shift)


def case_support(times, p, release, disruptive, down, threshold):
    """Postseal use of HDF5: strict lead time; no safe window after first false alarm."""
    keep = (down-times)*1000.>=-1e-6 if disruptive else np.ones(len(times),bool)
    hits = np.flatnonzero(keep & (p>=threshold))
    first = int(hits[0]) if len(hits) else None
    lead = round(float((down-times[first])*1000.),9) if disruptive and first is not None else None
    late = bool(disruptive and (lead is None or lead<=30.))
    fp = bool(not disruptive and first is not None)
    at = np.round((down-times[release])*1000.,9) if disruptive else np.full(len(release),np.nan)
    positive = (at>30.) & (at<=125.)
    safe = (release<=first) if fp else np.full(len(release),not disruptive)
    return dict(first=first,first_lead_ms=lead,late=late,fp=fp,
                pre_feedback_fp=bool(fp and not safe.any()),positive=positive,safe=safe)


def _npz(path, **arrays):
    with path.open('xb') as f:
        np.savez_compressed(f,**arrays)


def observe(reference, out):
    started = time.monotonic()
    lock = lag.load_inputs()
    p, seal = [read(reference/k) for k in ('protocol.json','seal.json')]
    bp, bs = [read(BRIDGE/k) for k in ('protocol.json','seal.json')]
    coords = BRIDGE/'decoders.npz'
    if (p['order']!=lock['ids'] or bp['order']!=lock['ids'] or Path(bp['reference'])!=reference
            or bp['reference_seal']!=sha(reference/'seal.json') or p['source_sha256']!=bp['source_sha256']
            or sha(coords)!=bs['assets'][coords.name] or sha(BRIDGE/'protocol.json')!=bs['assets']['protocol.json']):
        raise ValueError('the fixed 360 and previous observation coordinates must agree')
    with np.load(coords,allow_pickle=False) as f:
        center, scale = f['center'],f['scale']
    out.mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_COINCIDENT_INNOVATION_SHADOW_NOT_ALARMS',
        reference=str(reference),reference_seal=sha(reference/'seal.json'),order=lock['ids'],
        producer_sha256=sha(Path(__file__)),coordinate_sha256=sha(coords),startup=lag.STARTUP,
        source_sha256=p['source_sha256'],main_threshold=p['calibration']['0.2'],original=p['original'],
        release=FIRST,stride=STRIDE,horizon=H,families=FAMILIES,relations=PAIR_NAMES,
        calipers=CALIPER.tolist(),block_shots=BLOCK_SHOTS,
        release_contract='first effect at q+26; only complete 25 arrived samples and earlier context',
        marginal='per-family envelope mean, peak and slope; normalization uses preceding 125 points',
        relation='pairwise cosines of detrended five-bin absolute motion envelopes',
        control='roll already computed residual envelopes by [0,1,2,3] bins separately; keep all '
                'marginal statistics and observed support exactly',
        labels='no outcomes in observation; only prior sealed blocks used in supervised paired probes'))
    rows, witnesses = [], {}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed outcome access')):
        for rank,sid in enumerate(lock['ids'],1):
            name = f'{rank:04d}_{sid}.npz'
            raw_path, old = io23.path_for('EAST',sid),reference/'shots'/name
            if sha(old)!=seal['curve_sha256'][name] or sha(raw_path)!=lock['hashes'][str(sid)]:
                raise ValueError('raw signal or fixed curve is not the sealed reference')
            if rank<=lag.STARTUP:
                continue  # Cold shots only define the inherited parent; never teach the new probe.
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            mapped = lag.prepare_actual(raw,rank,lock['mapping'])
            signals = shot_coordinates(mapped,center,scale)
            feature = observed_features(*signals)
            r = feature['release']
            with np.load(old,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['times'],raw['times'])
                if not np.isin(r,f['release']).all():
                    raise ValueError('observed frame outside the inherited feedback clock')
                z = logit(f['fixed'][r].astype(float).clip(1e-6,1.-1e-6))
            x = np.c_[z,feature['context'],feature['single']]
            row = dict(rank=np.full(len(r),rank),shot=np.full(len(r),sid),time=raw['times'][r],
                       marginal=x, match=x[:,:len(CALIPER)],
                       **{k:feature[k] for k in ('release','aligned','shifted')})
            if not all(np.isfinite(v).all() for v in row.values()):
                raise FloatingPointError('nonfinite synchronous evidence')
            if rank in (21,120,360):
                end = min(410,len(raw['times']))
                short = observed_features(*[a[:end] for a in signals])
                for k,a in short.items():
                    np.testing.assert_array_equal(a,feature[k][r<end])
                witnesses[str(sid)] = end
            rows.append(row)
            if rank%85==0 or rank==360:
                print(dict(stage='seal_observations',rank=rank),flush=True)
    _npz(out/'frames.npz',**excursion._stack(rows))
    write(out/'observation_seal.json',dict(prefix_witnesses=witnesses,
        assets={k:sha(out/k) for k in ('frames.npz','protocol.json','executed_probe.py')},
        seconds=time.monotonic()-started))


def _designs(rows):
    x = rows['marginal']
    return dict(marginal=x,aligned=np.c_[x,rows['aligned']],shifted=np.c_[x,rows['shifted']])


def _pair(rows,positive,safe):
    p,s = [np.flatnonzero(v) for v in (positive,safe)]
    pairs = excursion.disjoint_pairs(rows['match'][p],rows['match'][s],rows['shot'][p],rows['shot'][s],CALIPER)
    return p[pairs[:,0]],s[pairs[:,1]]


def _predictions(rows):
    return dict(fixed=rows['marginal'][:,0],aligned_mean=rows['aligned'].mean(1),
                shifted_mean=rows['shifted'].mean(1),
                **{f'{arm}_{pair}':rows[arm][:,k] for arm in ('aligned','shifted')
                   for k,pair in enumerate(PAIR_NAMES)})


def _evaluate(rows,positive,safe,scores):
    p,s = _pair(rows,positive,safe)
    diff = {k:v[p]-v[s] for k,v in scores.items()}
    return dict(pairs=len(p),positive_shots=len(np.unique(rows['shot'][positive])),
                safe_shots=len(np.unique(rows['shot'][safe])),
                source_indices=np.stack((p,s),1).tolist(),
                matched_shots=np.stack((rows['shot'][p],rows['shot'][s]),1).tolist(),
                matched_release=np.stack((rows['release'][p],rows['release'][s]),1).tolist(),
                ordering={k:dict(correct=int((v>0.).sum()),ties=int((v==0.).sum()),
                    median=float(np.median(v)) if len(v) else None) for k,v in diff.items()})


def _reveal(p, reference_seals, rows, lo, hi):
    indices, truth = [], []
    d, late, fp, post = [np.zeros(len(rows['rank']),bool) for _ in range(4)]
    for rank in range(lo,hi):
        sid = p['order'][rank-1]
        i = np.flatnonzero(rows['rank']==rank)
        old = Path(p['reference'])/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(old)!=reference_seals[old.name]:
            raise ValueError('changed parent curve at outcome reveal')
        y,t = io23.read_outcome('EAST',sid)
        with np.load(old,allow_pickle=False) as f:
            np.testing.assert_array_equal(f['times'][rows['release'][i]],rows['time'][i])
            r = case_support(f['times'],f['fixed'],rows['release'][i],y,t,
                             p['original'] if rank<=p['startup'] else p['main_threshold'])
        d[i], late[i] = r['positive'],r['positive'] & r['late']
        post[i], fp[i] = r['safe'],r['safe'] & r['fp']
        indices.extend(i.tolist())
        truth.append(dict(rank=rank,shot=sid,disruptive=y,
                          **{k:r[k] for k in ('first','first_lead_ms','late','fp','pre_feedback_fp')},
                          any_timely_support=bool(r['positive'].any())))
    keep = np.array(indices,dtype=int)
    summary = dict(**rows,disrupt=d,late=late,fp=fp,safe=post)
    return {k:v[keep] for k,v in summary.items()},truth


def score(out):
    started = time.monotonic()
    p, seal = [read(out/k) for k in ('protocol.json','observation_seal.json')]
    if any(sha(out/k)!=h for k,h in seal['assets'].items()) or sha(out/'executed_probe.py')!=p['producer_sha256']:
        raise ValueError('changed observation archive')
    ref = Path(p['reference'])
    if sha(ref/'seal.json')!=p['reference_seal']:
        raise ValueError('changed reference seal')
    refseal = read(ref/'seal.json')['curve_sha256']
    with np.load(out/'frames.npz',allow_pickle=False) as f:
        rows = dict(f)
    _,truth = _reveal(p,refseal,rows,1,lag.STARTUP+1)
    history, runs = [], []
    for lo in range(lag.STARTUP+1,len(p['order'])+1,BLOCK_SHOTS):
        hi = min(lo+BLOCK_SHOTS,len(p['order'])+1)
        block = (rows['rank']>=lo) & (rows['rank']<hi)
        current = {k:v[block] for k,v in rows.items()}
        logits = _predictions(current)
        fitted = None
        if history:
            previous = excursion._stack(history)
            if previous['rank'].max()>=lo:
                raise ValueError('prospective probe received future outcomes')
            i,j = _pair(previous,previous['late'],previous['fp'])
            if len(i):
                fitted = dict(pairs=len(i),shots=np.c_[previous['shot'][i],previous['shot'][j]].tolist(),
                              ranks=np.c_[previous['rank'][i],previous['rank'][j]].tolist())
                designs = _designs(current)
                for k,x in _designs(previous).items():
                    coef = excursion.pair_fit(x[i],x[j])
                    logits[k] = designs[k]@coef
                    fitted[k] = coef.tolist()
        # Current raw labels are accessed only after the block's predictions are closed and hashed.
        f = out/f'block_{lo:03d}_predictions.npz'
        _npz(f,rank=current['rank'],shot=current['shot'],release=current['release'],**logits)
        digest = sha(f)
        observed,case = _reveal(p,refseal,rows,lo,hi)
        np.testing.assert_array_equal(observed['release'],current['release'])
        tests = dict(all=_evaluate(observed,observed['disrupt'],observed['safe'],logits),
                     hard=_evaluate(observed,observed['late'],observed['fp'],logits))
        runs.append(dict(first_rank=lo,last_rank=hi-1,fit=fitted,
                         forecasts=f.name,sha256=digest,tests=tests))
        history.append(observed)
        truth.extend(case)
        print(dict(stage='chronological_probe',start_rank=lo,fit_pairs=fitted['pairs'] if fitted else 0,
                   hard_pairs=tests['hard']['pairs']),flush=True)
    data = excursion._stack(history)
    if len(truth)!=360 or sum(r['disruptive'] for r in truth)!=180:
        raise ValueError('raw truth counts differ from the fixed 360')
    fp = [r for r in truth if r['fp']]
    missed = [r for r in truth if r['late']]
    counts = dict(static_timely=180-len(missed),static_fp=len(fp),
        fp_cold=[r['shot'] for r in fp if r['rank']<=lag.STARTUP],
        fp_before_first_feedback=[r['shot'] for r in fp if r['rank']>lag.STARTUP and r['pre_feedback_fp']],
        fp_with_feedback=[r['shot'] for r in fp if r['rank']>lag.STARTUP and not r['pre_feedback_fp']],
        late_suffix=sum(r['rank']>lag.STARTUP for r in missed),
        late_suffix_with_timely_support=sum(r['rank']>lag.STARTUP and r['any_timely_support'] for r in missed))
    controls = _predictions(data)
    global_tests = dict(all=_evaluate(data,data['disrupt'],data['safe'],controls),
                        hard=_evaluate(data,data['late'],data['fp'],controls))
    write(out/'result.json',dict(status='COMPLETED_COINCIDENT_INNOVATION_SHADOW_NOT_ALARMS',
        raw_counts=counts,shots=len(truth),frames=len(data['rank']),blocks=runs,
        global_postseal_pairs=global_tests,per_shot_clock=truth,seconds=time.monotonic()-started))
    print(dict(clock=counts,global_hard=global_tests['hard']['pairs']),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('observe','score'))
    parser.add_argument('out',type=Path)
    parser.add_argument('--reference',type=Path,default=bridge.REFERENCE)
    a = parser.parse_args()
    observe(a.reference.resolve(),a.out) if a.stage=='observe' else score(a.out)
