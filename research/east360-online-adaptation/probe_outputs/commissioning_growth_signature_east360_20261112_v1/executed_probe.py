"""Audit whether ordered acceleration in arrived measurements teaches warning direction."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit

from . import commissioning_arrival_paths_20261111 as arrival
from . import commissioning_coincident_innovation_20261111 as joint
from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_excursion_order_20261111 as excursion
from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .commissioning_transition_source_20261110 import shot_coordinates
from .history_gradient_metric_20261102.run import read, sha, write


REFERENCE = io23.ROOT/'probe_outputs/commissioning_bias_dynamics_east360_20261112_v1'
FIRST, STRIDE, H = joint.FIRST, joint.STRIDE, joint.H
GROUPS, FAMILIES = excursion.GROUPS, excursion.FAMILIES
FLOOR = excursion.FLOOR
KNOTS = np.arange(-16, 25, 8)
CALIPER = np.r_[joint.CALIPER, .35]
PARENT = 'bias_only'
MODELS = ('unordered', 'true_order', 'wrong_order')


def permute_steps(step, shift):
    # The same reordering applies across all diagnostics, so the steps' joint
    # distribution, cross-family coincidence, and terminal displacement survive.
    return step[np.arange(len(step))[:, None], (np.arange(3)-shift[:, None]) % 3]


def signature(before, step, shift):
    common, true, wrong = [], [], []
    for g in range(len(FAMILIES)):
        v, d = before[:, GROUPS==g], step[:, :, GROUPS==g]
        amplitude = np.sqrt(np.square(v).mean(1))
        energy = np.sqrt(np.square(d).mean((1, 2)))
        displacement = np.sqrt(np.square(d.sum(1)).mean(1))
        parallel = (d*v[:, None]).mean(2)
        parallel /= (np.maximum(amplitude, FLOOR)*np.maximum(energy, FLOOR))[:, None]
        common.append(np.c_[np.log1p(amplitude/FLOOR), np.log1p(energy/FLOOR),
                            np.log1p(displacement/FLOOR), np.sort(parallel, axis=1)])
        shifted = permute_steps(parallel, shift)
        true.append(.5*(parallel[:, 2]-parallel[:, 0]))
        wrong.append(.5*(shifted[:, 2]-shifted[:, 0]))
    return np.column_stack(common), np.column_stack(true), np.column_stack(wrong)


def observed(z, mask, release, rank):
    q = release-H-1
    arrive = q[:, None]+np.arange(1, H+1)
    knots = q[:, None, None]+KNOTS[None, :, None]+np.arange(-4, 1)
    good = (mask[arrive].all((1, 2)) & mask[knots].all((1, 2, 3))
            & np.isfinite(z[arrive]).all((1, 2)) & np.isfinite(z[knots]).all((1, 2, 3)))
    k = z[knots[good]].mean(2, dtype=np.float64)
    # A prior 16-sample direction, then three equal 8-sample increments.
    before, step = .5*(k[:, 2]-k[:, 0]), np.diff(k[:, 2:], axis=1)
    shift = 1 + ((rank+(release[good]-FIRST)//STRIDE) % 2)
    arrays = signature(before, step, shift)
    result = dict(usable=good)
    for name,a in zip(('invariant','true','wrong'),arrays,strict=True):
        result[name] = np.zeros((len(release), a.shape[1]), dtype=np.float32)
        result[name][good] = a
    return result


def next_estimate(release, usable, true, wrong):
    # At the next issue q=r-26, r-32 is already known. Current validity
    # only selects which already-issued forecasts can be scored.
    previous = np.flatnonzero((np.diff(release)==STRIDE) & usable[:-1])
    assert np.all(release[previous]<=release[previous+1]-H-1)
    ready = np.zeros(len(release), bool)
    ready[previous+1] = True
    t,w = np.zeros_like(true), np.zeros_like(wrong)
    t[previous+1],w[previous+1] = true[previous],wrong[previous]
    return ready & usable, dict(zero=np.zeros_like(true), past_true=t, past_wrong=w)


def observe(out):
    began = time.monotonic()
    lock = lag.load_inputs()
    src, frames = arrival._prior(arrival.REFERENCE)
    p,s = [read(REFERENCE/k) for k in ('protocol.json','seal.json')]
    coord = joint.BRIDGE/'decoders.npz'
    if (lock['ids']!=p['order'] or src['order']!=p['order'] or src['startup']!=p['startup']
            or len(s['curve_sha256'])!=360 or sha(coord)!=src['coordinate_sha256']
            or src['source_sha256']!=p['source_sha256']
            or src['main_threshold']!=p['calibration']['0.2'] or src['original']!=p['original']):
        raise ValueError('prior cohort, source, coordinate or threshold differs')
    with np.load(coord, allow_pickle=False) as f:
        center,scale = f['center'],f['scale']
    out.mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json', dict(scope='EAST360_DIRECTIONAL_ACCELERATION_PROBE_NOT_ALARMS',
        parent=str(REFERENCE), parent_seal=sha(REFERENCE/'seal.json'),
        parent_result=sha(REFERENCE/'per_shot.json'), parent_curve=PARENT,
        prior=str(arrival.REFERENCE), prior_seal=sha(arrival.REFERENCE/'observation_seal.json'),
        order=p['order'], startup=p['startup'], source_sha256=p['source_sha256'],
        main_threshold=p['calibration']['0.2'], original=p['original'],
        release=FIRST, stride=STRIDE, horizon=H, knot_points=KNOTS.tolist(),
        knot_mean_points=5, family_order=FAMILIES, floor=FLOOR, caliper=CALIPER.tolist(),
        block_shots=joint.BLOCK_SHOTS,
        null='per-row deterministic alternating 1/2 cyclic roll of joint 3 increments',
        fit='same disjoint broad positives and safe pairs in previous completed blocks only',
        clock='complete 25 samples, score from q+26; next curvature guess issued at q+32; '
              'hold last available within-shot estimate only if previous clock is contiguous'))
    rows,witnesses = [],{}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed outcome access')):
        for rank,sid in enumerate(lock['ids'], 1):
            if rank<=p['startup']:
                continue
            old = REFERENCE/'shots'/f'{rank:04d}_{sid}.npz'
            path = io23.path_for('EAST',sid)
            if sha(old)!=s['curve_sha256'][old.name] or sha(path)!=lock['hashes'][str(sid)]:
                raise ValueError('raw or parent curve hash differs from locked replay')
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            signal = lag.prepare_actual(raw,rank,lock['mapping'])
            z,m,_,_ = shot_coordinates(signal,center,scale)
            i = frames['rank']==rank
            r = frames['release'][i]
            with np.load(old,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['times'],raw['times'])
                np.testing.assert_array_equal(f['release'],r)
                parent = logit(f[PARENT][r].astype(float).clip(1e-6,1.-1e-6))
            np.testing.assert_array_equal(raw['times'][r],frames['time'][i])
            got = observed(z,m,r,rank)
            if rank in (21,120,360):
                stop = min(len(z),410)
                subset = observed(z[:stop],m[:stop],r[r<stop],rank)
                for k,v in subset.items():
                    np.testing.assert_array_equal(v,got[k][r<stop])
                witnesses[str(sid)] = stop
            eligible, guesses = next_estimate(r,got['usable'],got['true'],got['wrong'])
            previous_losses = np.stack([np.square(guess-got['true']).mean(1)
                                        for guess in guesses.values()], 1)
            age = np.log1p(r)
            marginal = np.c_[parent,frames['marginal'][i,1:],age]
            rows.append(dict(rank=frames['rank'][i], shot=frames['shot'][i], release=r,
                time=frames['time'][i], marginal=marginal, match=marginal[:, np.r_[
                    np.arange(len(joint.CALIPER)),marginal.shape[1]-1]],
                forecast_eligible=eligible, forecast_loss=previous_losses, **got))
            if rank%85==0 or rank==360:
                print(dict(stage='seal_ordered_acceleration',rank=rank),flush=True)
    data = excursion._stack(rows)
    if not all(np.isfinite(v).all() for v in data.values()):
        raise FloatingPointError('nonfinite observation')
    joint._npz(out/'frames.npz',**data)
    write(out/'seal.json',dict(prefix_witnesses=witnesses,
        seconds=time.monotonic()-began,
        assets={k:sha(out/k) for k in ('frames.npz','protocol.json','executed_probe.py')}))


def pairs(rows, positive, safe):
    a,b = (np.flatnonzero(rows['usable'] & valid) for valid in (positive,safe))
    chosen = excursion.disjoint_pairs(rows['match'][a],rows['match'][b],
                                      rows['shot'][a],rows['shot'][b],CALIPER)
    return a[chosen[:,0]],b[chosen[:,1]]


def comparisons(rows, positive, safe, scores):
    a,b = pairs(rows,positive,safe)
    gap = {k:v[a]-v[b] for k,v in scores.items()}
    return dict(pairs=len(a),positive_shots=len(np.unique(rows['shot'][positive & rows['usable']])),
        safe_shots=len(np.unique(rows['shot'][safe & rows['usable']])),
        shots=np.c_[rows['shot'][a],rows['shot'][b]].tolist(),
        releases=np.c_[rows['release'][a],rows['release'][b]].tolist(),
        ordering={k:dict(correct=int((v>0.).sum()),ties=int((v==0.).sum()),
                         median=float(np.median(v)) if len(v) else None) for k,v in gap.items()})


def designs(rows):
    common = np.c_[rows['marginal'],rows['invariant']]
    return dict(unordered=common,true_order=np.c_[common,rows['true']],
                wrong_order=np.c_[common,rows['wrong']])


def reveal(p, ref, sha_by_name, rows, lo, hi):
    i = np.flatnonzero((rows['rank']>=lo)&(rows['rank']<hi))
    data = {k:v[i] for k,v in rows.items()}
    data.update({k:np.zeros(len(i),bool) for k in ('positive','safe','late','miss','fp')})
    records = []
    for rank in range(lo,hi):
        sid = p['order'][rank-1]
        file = ref/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(file)!=sha_by_name[file.name]:
            raise ValueError('reference curve changed before reveal')
        disrupt,down,times = hdf._raw_truth(sid)
        at = data['rank']==rank
        r = data['release'][at]
        with np.load(file,allow_pickle=False) as f:
            np.testing.assert_array_equal(f['times'],times)
            prob = f[p['parent_curve']]
        np.testing.assert_array_equal(times[r],data['time'][at])
        result = joint.case_support(times,prob,r,disrupt,down,
                                    p['original'] if rank<=p['startup'] else p['main_threshold'])
        for k in ('positive','safe'):
            data[k][at] = result[k]
        data['late'][at] = result['positive'] & result['late'] & (result['first'] is not None)
        data['miss'][at] = result['positive'] & (result['first'] is None)
        data['fp'][at] = result['safe'] & result['fp']
        records.append(dict(shot=sid, disruptive=disrupt, down_s=down, alarm=result['first'] is not None,
                            lead_ms=result['first_lead_ms'], timely=disrupt and not result['late']))
    return data,records


def forecast_audit(rows, use):
    # Report means over shots, not pseudo-replicated windows.
    keep = use & rows['forecast_eligible']
    ranks = rows['rank'][keep]
    shot_means = np.asarray([rows['forecast_loss'][keep][ranks==r].mean(0,dtype=float)
                            for r in np.unique(ranks)]).reshape(-1,3)
    keys = ('zero','past_true','past_wrong')
    answer = dict(windows=int(keep.sum()),shots=len(shot_means))
    if len(shot_means):
        answer.update(loss=dict(zip(keys,shot_means.mean(0).tolist(),strict=True)),
            shots_better_than_zero=int((shot_means[:,1]<shot_means[:,0]).sum()),
            shots_better_than_wrong=int((shot_means[:,1]<shot_means[:,2]).sum()))
    return answer


def score(out):
    began = time.monotonic()
    p,s = [read(out/k) for k in ('protocol.json','seal.json')]
    ref = Path(p['parent'])
    if (any(sha(out/k)!=h for k,h in s['assets'].items())
            or sha(ref/'seal.json')!=p['parent_seal']
            or sha(ref/'per_shot.json')!=p['parent_result']):
        raise ValueError('sealed observations or parent outputs changed')
    refs = read(ref/'seal.json')['curve_sha256']
    with np.load(out/'frames.npz',allow_pickle=False) as f:
        rows = dict(f)
    _,records = reveal(p,ref,refs,rows,1,p['startup']+1)
    history, blocks = [],[]
    tests = (('all','positive','safe'),('late_vs_fp','late','fp'),('miss_vs_fp','miss','fp'))
    for start in range(p['startup']+1,len(p['order'])+1,p['block_shots']):
        stop = min(start+p['block_shots'],len(p['order'])+1)
        at = (rows['rank']>=start)&(rows['rank']<stop)
        current = {k:v[at] for k,v in rows.items()}
        scores = dict(parent=current['marginal'][:,0], raw_true=current['true'].mean(1),
                      raw_wrong=current['wrong'].mean(1))
        fit = None
        if history:
            earlier = excursion._stack(history)
            if earlier['rank'].max()>=start:
                raise ValueError('training saw the current block')
            a,b = pairs(earlier,earlier['positive'],earlier['safe'])
            if len(a):
                fit = dict(n=len(a),ranks=np.c_[earlier['rank'][a],earlier['rank'][b]].tolist())
                targets = designs(current)
                for k,x in designs(earlier).items():
                    coeff = excursion.pair_fit(x[a],x[b])
                    scores[k] = targets[k]@coeff
                    fit[k] = coeff.tolist()
        path = out/f'block_{start:03d}_predictions.npz'
        joint._npz(path,rank=current['rank'],release=current['release'],**scores)
        digest = sha(path)
        released,shot_result = reveal(p,ref,refs,rows,start,stop)
        records.extend(shot_result)
        history.append(released)
        result = {k:comparisons(released,released[a],released[b],scores) for k,a,b in tests}
        blocks.append(dict(start=start,stop=stop-1,fit=fit,prediction=path.name,sha256=digest,tests=result))
        print(dict(stage='score_sealed_block',start=start,fit=fit['n'] if fit else 0,
                   tests={k:v['pairs'] for k,v in result.items()}),flush=True)
    old = read(ref/'per_shot.json')['0.2'][p['parent_curve']]
    if records!=old or len(records)!=360 or sum(r['disruptive'] for r in records)!=180:
        raise ValueError('raw HDF5 metadata or score differs from the completed parent replay')
    r = excursion._stack(history)
    raw = dict(parent=r['marginal'][:,0], true=r['true'].mean(1),wrong=r['wrong'].mean(1))
    direct = {k:comparisons(r,r[a],r[b],raw) for k,a,b in tests}
    forecast = {k:forecast_audit(r,use) for k,use in
        (('all',np.ones(len(r['rank']),bool)),('timely_positive',r['positive']),('late',r['late']),
         ('miss',r['miss']),('before_first_fp',r['fp']))}
    write(out/'result.json',dict(status='COMPLETED_ACCELERATION_PROBE_NOT_ALARMS',raw_verified_shots=len(records),
        parent=dict(timely=sum(r['timely'] for r in records),
                    false_alarm=sum(r['alarm'] and not r['disruptive'] for r in records)),
        releases=len(r['rank']),usable=int(r['usable'].sum()),direct=direct,
        direct_next_forecast=forecast,blocks=blocks,seconds=time.monotonic()-began))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    observe(args.out.resolve())
    score(args.out.resolve())
