"""Test cross-family observability on sealed EAST360, without changing alarms."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch

from . import commissioning_coincident_innovation_20261111 as clock_score
from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_eos_replay_20261111 as replay
from . import commissioning_excursion_order_20261111 as excursion
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_response_bridge_20261111 as ridge
from . import io23, simple_data
from .commissioning_transition_data_20261110 import _window_mean
from .commissioning_transition_source_20261110 import shot_coordinates
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest


GROUP = excursion.GROUPS
FAMILIES = len(excursion.FAMILIES)
BINS, H = 5, lag.H
PARENT_ARM = 'signed_path_true'
METHODS = ('unadjusted', 'marginal', 'aligned', 'shifted')
LOCAL = ('local_bias', 'local_aligned', 'local_shifted')
HISTORY = 8
CONTROLS = ('level', 'ramp')
CALIPERS = np.r_[excursion.CALIPERS[:4], .15, .20]
SEED, BOOTSTRAPS = 131, 2000


def ready(q, observed, length):
    return np.flatnonzero(observed.all((1, 2)) & (q+H+1 < length))


def shift_groups(e):
    """Remove cross-family phase agreement, not per-channel magnitude or group membership."""
    y = np.empty_like(e)
    for g in range(FAMILIES):
        use = GROUP == g
        y[..., use] = np.roll(e[..., use], g+1, axis=1)
    return y


def fit_observer(e, shot):
    _, inv, counts = np.unique(shot, return_inverse=True, return_counts=True)
    weight = 1./counts[inv]/len(counts)
    mean = np.einsum('n,nbc->bc', weight, e)
    scale = np.maximum(np.sqrt(np.einsum('n,nbc->c', weight, (e-mean)**2)/BINS), ridge.FLOOR)
    u = (e-mean)/scale
    models = []
    for g in range(FAMILIES):
        held = GROUP == g
        models.append(ridge.fit_ridge(u[..., ~held].reshape(len(e), -1), u[..., held], shot))
    return mean, scale, models


def observer(e, fit):
    mean, scale, models = fit
    u = (e-mean)/scale
    predictions = {k:np.zeros_like(u) for k in METHODS[2:]}
    for g, model in enumerate(models):
        held = GROUP == g
        center, spread, average, coef = model
        # The target family's new block is absent from both design matrices.
        for k, v in zip(METHODS[2:], (u, shift_groups(u)), strict=True):
            x = v[..., ~held].reshape(len(e), -1)
            pred = ((x-center)/spread)@coef+average
            predictions[k][..., held] = pred.reshape(len(e), BINS, int(held.sum()))
    return dict(unadjusted=np.zeros_like(e), marginal=np.broadcast_to(mean, e.shape),
                **{k:mean+scale*v for k,v in predictions.items()})


def local_observer(e, release, fit):
    """Fit an offset and a cross-family covariance using only earlier arrived blocks."""
    mean, scale, _ = fit
    u = (e-mean)/scale
    i = np.arange(len(e))
    lo = np.searchsorted(release, release-HISTORY*lag.STRIDE)
    # One zero-residual reference block shrinks both estimates toward the cold bias.
    count = (i-lo+1).astype(float)
    cumulative = np.concatenate((np.zeros((1,*u.shape[1:])),np.cumsum(u,axis=0)))
    offset = (cumulative[i]-cumulative[lo])/count[:,None,None]
    gram = np.einsum('tbi,tbj->tij',u,u)
    gram = np.concatenate((np.zeros((1,*gram.shape[1:])),np.cumsum(gram,axis=0)))
    cov = ((gram[i]-gram[lo])/count[:,None,None]
           - np.einsum('tbi,tbj->tij',offset,offset))/BINS
    centered = u-offset
    shifted = shift_groups(centered)
    aligned, control = offset.copy(), offset.copy()
    for g in range(FAMILIES):
        held = GROUP==g
        xcov = cov[:,~held][:,:,~held]+ridge.RIDGE*np.eye(int((~held).sum()))
        coef = np.linalg.solve(xcov,cov[:,~held][:,:,held])
        aligned[...,held] += centered[...,~held]@coef
        control[...,held] += shifted[...,~held]@coef
    return {k:mean+scale*v for k,v in zip(LOCAL,(offset,aligned,control),strict=True)}


def past_gates(loss, release):
    """Choose an estimator before evaluating the held family's current block."""
    i = np.arange(len(release))
    lo = np.searchsorted(release,release-HISTORY*lag.STRIDE)
    a = np.log1p(loss[:,[1,4,5,6]].astype(float))
    sums = np.concatenate((np.zeros((1,4,FAMILIES)),np.cumsum(a,axis=0)))
    prior = sums[i]-sums[lo]
    enough = (i-lo>=2)[:,None]
    local = enough & (prior[:,1]<prior[:,0])
    allowed = enough & (prior[:,2]<np.where(local,prior[:,1],prior[:,0])) & (prior[:,2]<prior[:,3])
    return np.where(local,4,1), allowed


def losses(e, predictions, scale):
    return {k:np.stack([np.square((e[..., GROUP==g]-v[..., GROUP==g])/scale[GROUP==g])
                        .mean((1, 2)) for g in range(FAMILIES)], 1)
            for k,v in predictions.items()}


def synthetic_localization(e, fit):
    """One planted group at a time, balanced signs and unit standardized RMS."""
    _, scale, _ = fit
    shapes = (np.ones(BINS), np.arange(1., BINS+1.))
    result = {}
    for mode, shape in zip(CONTROLS, shapes, strict=True):
        shape = shape / np.sqrt(np.mean(shape**2))
        result[mode] = {k:np.zeros((len(e), FAMILIES), np.float32) for k in ('marginal', 'aligned')}
        for g in range(FAMILIES):
            take = GROUP == g
            for sign in (-1., 1.):
                changed = e.copy()
                changed[..., take] += sign*shape[None, :, None]*scale[take]
                pred = observer(changed, fit)
                for k,v in losses(changed, {k:pred[k] for k in result[mode]}, scale).items():
                    result[mode][k][:,g] += .5*(np.argmax(v, 1)==g)
    return result


@torch.no_grad()
def observed_blocks(signal, args, decoder):
    base, net, (center, scale, forecast, noise, _), *_ = args
    tr = lag.direct.track(net, signal, center, scale, np.empty(0, np.int64))
    i = ready(tr['q'], tr['observed'], tr['length'])
    q = tr['q'][i]
    release = q+H+1
    _, _, phys, pm = shot_coordinates(signal, center, scale)
    recent, n = _window_mean(phys, pm, release, -15, 16)
    before, b = _window_mean(phys, pm, release, -31, 16)
    valid = (n>=8).all(1) & (b[:,0]>=8)
    i, q, release = i[valid], q[valid], release[valid]
    context = np.tanh(np.c_[recent,4.*(recent[:,0]-before[:,0])])[valid]
    initial = lag.direct.base_forecast(tr, forecast)[i]
    measured = lag.direct.normalized_target(tr, noise)[i]
    pred = np.empty_like(initial)
    for start in range(0,len(i),128):
        sl = slice(start,start+128)
        x,m = lag._windows(base, signal, q[sl])
        h = lag._features(base, x, m, x.new_zeros((len(x), len(GROUP), len(GROUP))))
        pred[sl] = lag._predict(h, torch.from_numpy(initial[sl]).to(h.device), decoder).cpu().numpy()
    residual = (measured-pred).reshape(len(i), BINS, H//BINS, len(GROUP)).mean(2)
    return dict(release=release, residual=residual, context=context)


def _record(row, fit):
    e = row['residual']
    loss = losses(e, observer(e,fit) | local_observer(e,row['release'],fit), fit[1])
    localization = synthetic_localization(e,fit)
    return dict(loss=np.stack([loss[k] for k in (*METHODS,*LOCAL)],1).astype(np.float32),
        injected=np.stack([np.stack([localization[mode][k] for k in ('marginal','aligned')],1)
                           for mode in CONTROLS],1))


def observe(ref, out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    p, _, _, args, decoder = replay._load(ref, lock)
    base, net = args[:2]
    frozen = [state_digest(k) for k in (base,net)]
    seal = read(ref/'seal.json')
    out.mkdir()
    copyfile(Path(__file__), out/'executed_probe.py')
    cold, rows, witnesses, trained = [], [], {}, []
    for rank, sid in enumerate(lock['ids'][:p['startup']], 1):
        f = ref/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(f)!=seal['curve_sha256'][f.name]:
            raise ValueError('preparation prediction must be sealed before its outcome is used')
        if not io23.read_outcome('EAST',sid)[0]:
            trained.append(sid)
    write(out/'protocol.json',dict(scope='EAST360_HELD_DIAGNOSTIC_OBSERVABILITY_NO_NEW_ALARMS',
        order=lock['ids'], startup=p['startup'], parent=str(ref), parent_seal=sha(ref/'seal.json'),
        source=p['source_sha256'], producer=sha(Path(__file__)), original=p['original'],
        threshold=p['calibration']['0.2'], parent_arm=PARENT_ARM, cold_safe_ids=trained,
        horizon=H, bins=BINS, stride=lag.STRIDE, first_effect=lag.FIRST_EFFECT,
        diagnostics=dict(zip(excursion.FAMILIES,
            [[io23.RAW23[c] for c,g in zip(lag.CHANNELS,GROUP,strict=True) if g==j]
             for j in range(FAMILIES)],strict=True)),
        fit='one total weight per cold safe shot; within preparation leave that shot out of '
            'all observer statistics; the inherited coordinate map is fixed',
        view='use the same locked suffix coordinate map in both diagnostic fits and '
             'held tests; no modification to archived cold or suffix risk',
        timing='direct H25 forecast from q; cross-family estimate at q+26 uses only '
               'other families already arrived after q; its own new family is used only for evaluation',
        controls='same-frame cyclic shift by 1 to 4 bins; level and ramp injection in one '
                 'group only, both signs and one normalized RMS unit',
        local_fit='at most the preceding eight release slots on the same shot; offset and '
                  'pooled cross-family covariance share one zero-residual reference block; '
                  'conditioning on real vs shifted current blocks uses the same weights',
        local_history=HISTORY, ridge=ridge.RIDGE,
        calipers=CALIPERS.tolist(), matching=['logit','Ip','Bt','short_Ip_change',
            'log_elapsed','tanh_residual_rms'], methods=(*METHODS,*LOCAL), seed=SEED))
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed outcome access')):
        for rank, sid in enumerate(lock['ids'], 1):
            path = ref/'shots'/f'{rank:04d}_{sid}.npz'
            if sha(path)!=seal['curve_sha256'][path.name] or sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError('fixed parent curve or diagnostic signal changed')
            if rank<=p['startup'] and sid not in trained:
                continue
            raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            signal = lag.prepare_actual(raw, max(rank,p['startup']+1), lock['mapping'])
            row = observed_blocks(signal,args,decoder)
            r = row['release']
            with np.load(path,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['times'],raw['times'])
                if rank>p['startup']:
                    np.testing.assert_array_equal(f['release'],r)
                score = logit(f[PARENT_ARM][r].astype(float).clip(1e-7,1.-1e-7))
            t = raw['times'][r]
            row.update(shot=np.full(len(r),sid),rank=np.full(len(r),rank),time=t,
                match=np.c_[score,row.pop('context'),np.log1p(t-raw['times'][0])])
            if not all(np.isfinite(v).all() for v in row.values()):
                raise ValueError('invalid diagnostic forecast frame')
            if rank in (21,120,360):
                end = min(len(raw['times']),330)
                prefix = {k:v[:end] for k,v in signal.items() if k in ('values','available')}
                short = observed_blocks(prefix,args,decoder)
                np.testing.assert_array_equal(short['release'],r[r<end])
                np.testing.assert_allclose(short['residual'],row['residual'][r<end],rtol=0.,atol=3e-6)
                witnesses[str(sid)] = end
            if rank <= p['startup']:
                cold.append(row)
            else:
                if rank==p['startup']+1:
                    c = excursion._stack(cold)
                    if set(c['shot']) != set(trained):
                        raise ValueError('an authorized preparation shot has no complete observation')
                    fit = fit_observer(c['residual'],c['shot'])
                    for v in cold:
                        keep = c['shot']!=v['shot'][0]
                        v.update(_record(v,fit_observer(c['residual'][keep],c['shot'][keep])))
                row.update(_record(row,fit))
                rows.append(row)
            if rank%60==0 or rank==21:
                print(dict(stage='held_diagnostic',rank=rank,seconds=round(time.monotonic()-started,2)),
                      flush=True)
    if [state_digest(k) for k in (base,net)] != frozen:
        raise ValueError('a frozen model changed')
    records = excursion._stack([*cold,*rows])
    with (out/'frames.npz').open('xb') as f:
        np.savez_compressed(f,**{k:v for k,v in records.items() if k!='residual'})
    write(out/'seal.json',dict(prefix_witnesses=witnesses,seconds=time.monotonic()-started,
        assets={k:sha(out/k) for k in ('frames.npz','protocol.json','executed_probe.py')}))


def by_shot(x, shot):
    return np.stack([x[shot==sid].mean(0,dtype=np.float64) for sid in np.unique(shot)])


def summary(rows, keep):
    if not keep.any():
        return dict(frames=0,shots=0)
    shot = rows['shot'][keep]
    loss = by_shot(rows['loss'][keep],shot)
    injected = by_shot(rows['injected'][keep],shot).mean(0)
    means = loss.mean(0)
    sample = np.random.default_rng(SEED).integers(len(loss),size=(BOOTSTRAPS,len(loss)))

    def paired(a,b):
        gap = (loss[:,a]-loss[:,b]).mean(1)
        return dict(wins=int((gap>0.).sum()),mean=float(gap.mean()),median=float(np.median(gap)),
                    shot_resample_95=np.quantile(gap[sample].mean(1),[.025,.975]).tolist())

    methods = (*METHODS,*LOCAL)
    return dict(frames=int(keep.sum()), shots=len(loss),
        mse=dict(zip(methods, means.mean(1).tolist(),strict=True)),
        by_family={name:dict(zip(methods,means[:,j].tolist(),strict=True))
                   for j,name in enumerate(excursion.FAMILIES)},
        aligned_vs_marginal=paired(1,2), local_vs_fixed_bias=paired(1,4),
        local_aligned_vs_bias=paired(4,5), local_aligned_vs_shifted=paired(6,5),
        injected_correct={k:dict(zip(('marginal','aligned'),injected[j].mean(1).tolist(),strict=True))
                          for j,k in enumerate(CONTROLS)})


def matched(rows, disruptive, safe):
    loss = rows['loss'].mean(2,dtype=float)
    base = loss[:,1]
    x = np.c_[rows['match'],np.tanh(np.sqrt(base))]
    pos, neg = [np.flatnonzero(s) for s in (disruptive,safe)]
    pair = excursion.disjoint_pairs(x[pos],x[neg],rows['shot'][pos],rows['shot'][neg],CALIPERS)
    a,b = pos[pair[:,0]],neg[pair[:,1]]
    scores = dict(shared=(base-loss[:,2])/(base+loss[:,2]+1e-4),
        phase=(loss[:,3]-loss[:,2])/(loss[:,3]+loss[:,2]+1e-4),
        local_shared=(loss[:,4]-loss[:,5])/(loss[:,4]+loss[:,5]+1e-4),
        local_phase=(loss[:,6]-loss[:,5])/(loss[:,6]+loss[:,5]+1e-4),
        local_unexplained=np.sqrt(loss[:,5]), local_magnitude=np.sqrt(loss[:,4]),
        unexplained=np.sqrt(loss[:,2]),magnitude=np.sqrt(base),base_logit=x[:,0])
    return dict(pairs=len(a), matched_shots=np.c_[rows['shot'][a],rows['shot'][b]].tolist(),
        matched_release=np.c_[rows['release'][a],rows['release'][b]].tolist(),
        compare={k:dict(higher=int((v[a]>v[b]).sum()),tied=int((v[a]==v[b]).sum()),
                median_gap=float(np.median(v[a]-v[b])) if len(a) else None)
                 for k,v in scores.items()})


def _verified_inputs(out):
    p,s = (read(out/k) for k in ('protocol.json','seal.json'))
    ref = Path(p['parent'])
    if (any(sha(out/k)!=v for k,v in s['assets'].items()) or sha(ref/'seal.json')!=p['parent_seal']):
        raise ValueError('changed observation or parent seal')
    if p['methods']!=list((*METHODS,*LOCAL)):
        raise ValueError('use the sealed version of the scorer for this archive')
    with np.load(out/'frames.npz',allow_pickle=False) as f:
        rows = dict(f)
    return p, rows


def _reveal(p, rows):
    ref = Path(p['parent'])
    hashes = read(ref/'seal.json')['curve_sha256']
    old = read(ref/'per_shot.json')['0.2'][PARENT_ARM]
    shape = rows['rank'].shape
    suffix, near, late, safe, fp = (np.zeros(shape,bool) for _ in range(5))
    totals = dict(positive=0,timely=0,safe=0,fp=0,late_with_support=0,fp_with_support=0)
    for rank,(sid,previous) in enumerate(zip(p['order'],old,strict=True),1):
        path = ref/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(path)!=hashes[path.name]:
            raise ValueError('changed archived alarm')
        d,down,t = hdf._raw_truth(sid)
        i = np.flatnonzero(rows['rank']==rank)
        r = rows['release'][i]
        np.testing.assert_array_equal(rows['shot'][i],np.full(len(i),sid))
        np.testing.assert_array_equal(rows['time'][i],t[r])
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(t,f['times'])
            p0 = f[PARENT_ARM]
        limit = p['threshold'] if rank>p['startup'] else p['original']
        u = clock_score.case_support(t,p0,r,d,down,limit)
        if (sid!=previous['shot'] or d!=previous['disruptive'] or down!=previous['down_s']
                or (u['first'] is not None)!=previous['alarm'] or u['first_lead_ms']!=previous['lead_ms']
                or (d and not u['late'])!=previous['timely']):
            raise ValueError('raw outcome or first alarm does not reproduce')
        totals['positive']+=int(d)
        totals['safe']+=int(not d)
        totals['timely']+=int(previous['timely'])
        totals['fp']+=int(u['fp'])
        suffix[i] = rank>p['startup']
        near[i], late[i] = u['positive'],u['late'] & u['positive']
        safe[i],fp[i] = u['safe'],u['fp'] & u['safe']
        if rank>p['startup']:
            totals['late_with_support']+=int(late[i].any())
            totals['fp_with_support']+=int(fp[i].any())
    return totals, dict(suffix=suffix, near=near, late=late, safe=safe, fp=fp)


def score(out):
    p, rows = _verified_inputs(out)
    totals, masks = _reveal(p,rows)
    suffix, near, late, safe, fp = (masks[k] for k in ('suffix','near','late','safe','fp'))
    result = dict(status='POSTSEAL_HELD_DIAGNOSTIC_NO_ALARM_CHANGE', raw_shots=len(p['order']),
        archived_parent_counts=totals, coverage_shots=len(np.unique(rows['shot'][suffix])),
        summaries={k:summary(rows,v) for k,v in (('cold_safe_loo',~suffix),('all_suffix',suffix),
            ('timely_window',near & suffix),('late_window',late & suffix),
            ('safe_pre_first',safe & suffix),('false_alarm_pre_first',fp & suffix))},
        matching=dict(broad=matched(rows,near & suffix,safe & suffix),
                      hard=matched(rows,late & suffix,fp & suffix)))
    write(out/'result.json',result)
    print(dict(parent=totals, held={k:v.get('mse') for k,v in result['summaries'].items()},
               hard=result['matching']['hard']['compare']),flush=True)


def score_gate(out):
    p, rows = _verified_inputs(out)
    loss, release, shot = (rows[k] for k in ('loss','release','shot'))
    bias = np.ones_like(loss[:,0],dtype=np.int8)
    allow = np.zeros_like(bias,dtype=bool)
    for sid in np.unique(shot):
        k = np.flatnonzero(shot==sid)
        bias[k],allow[k] = past_gates(loss[k],release[k])
    trace = out/'past_gate_decisions.npz'
    with trace.open('xb') as f:
        np.savez_compressed(f,shot=shot,release=release,bias=bias,cross_family=allow)
    copyfile(Path(__file__),out/'executed_past_gate.py')
    digest = sha(trace)
    # Outcome access is restricted to scoring already written decisions.
    totals,m = _reveal(p,rows)
    suffix = m['suffix']
    names = ('fixed_bias','causal_bias_only','qualified_real','qualified_shifted')
    indices = (np.ones_like(bias),bias,np.where(allow,5,bias),np.where(allow,6,bias))
    err = np.stack([np.take_along_axis(loss,j[:,None],axis=1)[:,0] for j in indices],1)
    groups = {}
    for k, v in (('suffix',suffix),('late',suffix&m['late']),('near',suffix&m['near']),
                 ('actual_false_alarm',suffix&m['fp']),('safe_pre_alarm',suffix&m['safe'])):
        value = by_shot(err[v],shot[v]).mean(2)
        enabled = v & allow.any(1)
        groups[k] = dict(shots=len(value),frames=int(v.sum()),enabled_frames=int(enabled.sum()),
            enabled_shots=int(len(np.unique(shot[enabled]))),enabled_groups=int(allow[v].sum()),
            mse=dict(zip(names,value.mean(0).tolist(),strict=True)),
            real_beats_bias=int((value[:,2]<value[:,1]).sum()),
            real_beats_shifted=int((value[:,2]<value[:,3]).sum()))
    write(out/'past_gate_result.json',dict(status='CAUSAL_HELD_DIAGNOSTIC_GATE_NO_NEW_ALARMS',
        timing='choose per group from two or more earlier released blocks in preceding 8 slots',
        criterion='lower accumulated log1p MSE than chosen bias-only and shifted estimators',
        same_selection_for_real_and_shifted=True, trace_sha256=digest,
        executed_sha256=sha(Path(__file__)), parent_counts=totals, groups=groups))
    print(groups,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('observe','score','past_gate'))
    parser.add_argument('out',type=Path)
    a = parser.parse_args()
    if a.stage=='observe':
        observe(replay.REFERENCE,a.out)
    else:
        {'score':score,'past_gate':score_gate}[a.stage](a.out)
