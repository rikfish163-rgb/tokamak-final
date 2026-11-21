"""Separate the predictability of the next diagnostic excursion from its warning relevance."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch

from . import commissioning_eos_replay_20261111 as replay
from . import commissioning_excursion_order_20261111 as excursion
from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .commissioning_transition_data_20261110 import _window_mean
from .commissioning_transition_source_20261110 import shot_coordinates
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest


REFERENCE = io23.ROOT/'probe_outputs/commissioning_segment_teacher_east360_20261111_v2'
ARMS = ('hold', 'endpoints', 'ordered', 'reversed_order', 'hidden', 'hidden_ordered')
FIT = ('endpoints', 'ordered', 'hidden', 'hidden_ordered')
RIDGE, FLOOR = .10, .05
H, STRIDE, FIRST = lag.H, lag.STRIDE, lag.FIRST_EFFECT


def prefix_views(z, m, phys, pm):
    """Issue a second block forecast after observing only the first complete block."""
    issue = np.arange(FIRST, len(z), STRIDE, dtype=np.int64)
    q = issue-H-1
    means = []
    usable = np.ones(len(q), dtype=bool)
    for start, n in ((-127, 64), (-7, 8), (1, 8), (9, 8), (17, 9)):
        idx = q[:, None]+start+np.arange(n)
        usable &= m[idx].all((1, 2))
        means.append(z[idx].mean(1, dtype=np.float64))
    recent, n = _window_mean(phys, pm, issue, -15, 16)
    before, b = _window_mean(phys, pm, issue, -31, 16)
    usable &= (n >= 8).all(1) & (b[:, 0] >= 8)
    ref, start, *path = [a[usable] for a in means]
    same, ordered, amp = excursion.describe_excursion(ref, start, np.stack(path, 1))
    tail = path[-1]
    context = np.tanh(np.c_[recent, 4.*(recent[:, 0]-before[:, 0])])[usable]
    return dict(issue=issue[usable], same=np.c_[context, ref, start, tail, same], order=ordered,
                match=np.c_[context, amp], ref=ref, tail=tail)


def radial_response(ref, tail, future):
    """Per-family log relative distance to the same earlier reference, at every future point."""
    new, old = future-ref[:, None], tail-ref
    out = []
    for g in range(excursion.N_GROUPS):
        use = excursion.GROUPS == g
        a = np.square(new[:, :, use]).mean(2)+FLOOR**2
        b = np.square(old[:, use]).mean(1)+FLOOR**2
        out.append((.5*np.log(a/b[:, None])).clip(-3., 3.))
    return np.stack(out, 2)


def realized(z, m, views):
    """Select only complete next blocks; this is never used to select forecast issues."""
    q = views['issue']
    ready = np.flatnonzero(q+H+1 < len(z))
    idx = q[ready, None]+np.arange(1, H+1)
    ok = m[idx].all((1, 2))
    use, idx = ready[ok], idx[ok]
    return use, radial_response(views['ref'][use], views['tail'][use], z[idx])


def fit_ridge(x, y, shot):
    """One shot, one total weight. Return a direct H25 multivariate decoder."""
    _, group, n = np.unique(shot, return_inverse=True, return_counts=True)
    weight = 1./n[group]/len(n)
    center = weight @ x
    scale = np.sqrt(np.maximum(weight @ np.square(x-center), FLOOR**2))
    z = (x-center)/scale
    target = y.reshape(len(x), -1)
    mean = weight @ target
    cov = z.T*weight
    coef = np.linalg.solve(cov@z+RIDGE*np.eye(x.shape[1]), cov@(target-mean))
    return center, scale, mean, coef


def predict_ridge(x, fit):
    center, scale, mean, coef = fit
    return (((x-center)/scale)@coef+mean).reshape(len(x), H, excursion.N_GROUPS)


def _design(same, ordered, hidden):
    return dict(endpoints=same, ordered=np.c_[same, ordered], reversed_order=np.c_[same,-ordered],
                hidden=np.c_[same, hidden], hidden_ordered=np.c_[same, hidden, ordered])


def _load(ref, lock):
    p, seal = [read(ref/k) for k in ('protocol.json','seal.json')]
    parent = Path(p['reference'])
    gp, _, _, args, _ = replay._load(parent, lock)
    if (p['scope'] != 'EAST360_NATURAL_EOS_INTERNAL_SEGMENT_TEACHING_DEVELOPMENT'
            or p['order'] != lock['ids'] or p['source_sha256'] != gp['source_sha256']
            or p['calibration'] != gp['calibration'] or p['original'] != gp['original']
            or sha(parent/'seal.json') != p['reference_seal'] or len(seal['curve_sha256']) != 360):
        raise ValueError('the commissioned fixed risk reference must match the locked full flow')
    return p, seal, args


@torch.no_grad()
def _input(signal, base, z, m, phys, pm, fixed):
    views = prefix_views(z, m, phys, pm)
    issue = views['issue']
    if not len(issue):
        raise ValueError('no causal diagnostic support on this discharge')
    x, avail = lag._windows(base, signal, issue)
    h = lag._features(base, x, avail, x.new_zeros(len(x),len(lag.CHANNELS),len(lag.CHANNELS)))
    v = logit(np.clip(fixed[issue].astype(float), 1e-6, 1.-1e-6))
    views['match'] = np.c_[v, views['match']]
    return views, _design(np.c_[v, views['same']], views['order'], h.cpu().numpy())


def _response(designs, models):
    n = len(designs['endpoints'])
    by_arm = [np.zeros((n,H,excursion.N_GROUPS))]
    by_arm.extend(predict_ridge(designs[k],models['ordered' if k == 'reversed_order' else k])
                  for k in ARMS[1:])
    return np.stack(by_arm, 1).astype(np.float32)


def _npz(path, **data):
    with path.open('xb') as f:
        np.savez_compressed(f, **data)


def predict(ref, out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    p, seal, args = _load(ref, lock)
    base = args[0]
    frozen = state_digest(base)
    center, scale = [v.cpu().numpy() for v in (base.center,base.scale)]
    out.mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_DIAGNOSTIC_RESPONSE_SHADOW_NOT_ALARMS',
        reference=str(ref), reference_seal=sha(ref/'seal.json'), order=lock['ids'],
        producer_sha256=sha(Path(__file__)), source_sha256=p['source_sha256'], arms=ARMS,
        fit_scope='only diagnostic signals of the ended preparation 20, in the locked suffix coordinates',
        scored_scope='all held suffix discharges, no outcomes or held signals in fit',
        context=['frozen_logit','Ip','Bt','short_Ip_change','start_radius','terminal_radius'],
        calipers=excursion.CALIPERS.tolist(), families=excursion.FAMILIES,
        horizon=H, stride=STRIDE, first_issue=FIRST, ridge=RIDGE, floor=FLOOR,
        controls='same completed first-block endpoints, invariant moments and context; order adds signed '
                 'difference of the first two bins; reversed uses the exact fitted order head',
        target='direct 25 points x four diagnostic families; log radial change relative to prior '
               '64-point mean, not a disruption label; current feature never sees this later block'))
    training = {k:[] for k in FIT}
    y, train_id, rows, coverage, witnesses = [], [], [], [], {}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed outcome access')):
        for rank,sid in enumerate(lock['ids'],1):
            name = f'{rank:04d}_{sid}.npz'
            old, raw = ref/'shots'/name, io23.path_for('EAST',sid)
            if sha(old)!=seal['curve_sha256'][name] or sha(raw)!=lock['hashes'][str(sid)]:
                raise ValueError('locked raw data or risk trace changed')
            discharge = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            signal = lag.prepare_actual(discharge,max(rank,lag.STARTUP+1),lock['mapping'])
            coords = shot_coordinates(signal,center,scale)
            with np.load(old,allow_pickle=False) as f:
                np.testing.assert_array_equal(discharge['times'],f['times'])
                fixed = f['fixed']
            if rank<=lag.STARTUP:
                # Match the coordinate and risk readout used on the suffix, not the unaligned cold score.
                fixed = replay._track(signal,args)[2].sigmoid().cpu().numpy()
            views, design = _input(signal,base,*coords,fixed)
            coverage.append(dict(rank=rank,shot=sid,issue_count=len(views['issue'])))
            if rank<=lag.STARTUP:
                use, target = realized(*coords[:2],views)
                y.append(target)
                train_id.extend([sid]*len(use))
                for k in FIT:
                    training[k].append(design[k][use])
                coverage[-1]['fit_rows'] = len(use)
                if rank==lag.STARTUP:
                    if set(train_id)!=set(lock['ids'][:lag.STARTUP]):
                        raise ValueError('complete training support in the authorized preparation set required')
                    models = {k:fit_ridge(np.concatenate(a),np.concatenate(y),np.array(train_id))
                              for k,a in training.items()}
                    _npz(out/'decoders.npz', center=center, scale=scale, fit_ids=np.unique(train_id),
                        **{f'{k}_{name}':a for k,fit in models.items() for name,a in zip(
                            ('center','scale','mean','coef'),fit,strict=True)})
            else:
                preds = _response(design, models)
                if not np.isfinite(preds).all():
                    raise FloatingPointError('nonfinite trajectory forecast')
                row = dict(rank=np.full(len(preds),rank), shot=np.full(len(preds),sid),
                           **{k:views[k] for k in ('issue','match','ref','tail')}, prediction=preds)
                if rank in (21,120,360):
                    end = int(views['issue'][min(6,len(preds)-1)])+1
                    prefix = {k:v[:end] for k,v in signal.items() if k in ('values','available')}
                    short, short_x = _input(prefix,base,*[a[:end] for a in coords],fixed[:end])
                    mask = views['issue']<end
                    np.testing.assert_array_equal(short['issue'],views['issue'][mask])
                    np.testing.assert_allclose(_response(short_x,models),preds[mask],atol=2e-6,rtol=0.)
                    witnesses[str(sid)] = end
                rows.append(row)
            if rank in (20,21) or rank%80==0 or rank==360:
                print(dict(stage='prefix_only_forecast',rank=rank,seconds=round(time.monotonic()-started,2)),
                      flush=True)
    if state_digest(base)!=frozen:
        raise ValueError('frozen encoder changed')
    _npz(out/'forecasts.npz',**excursion._stack(rows))
    write(out/'seal.json',dict(shots=coverage,prefix_witnesses=witnesses,
        assets={k:sha(out/k) for k in ('decoders.npz','executed_probe.py','forecasts.npz','protocol.json')},
        seconds=time.monotonic()-started))


def _losses(y, predictions, ranks):
    if not len(y):
        return dict(rows=0, shots=0, arms={})
    _, ix, n = np.unique(ranks,return_inverse=True,return_counts=True)
    weight = (1./n[ix])/len(n)
    err = predictions-y[:, None]
    mae = np.einsum('rag,r->ag',np.abs(err).mean(2),weight)
    mse = np.einsum('rag,r->ag',np.square(err).mean(2),weight)
    return dict(rows=len(y), shots=len(n), arms={k:dict(mae=float(a.mean()),mse=float(b.mean()),
        by_family_mae=dict(zip(excursion.FAMILIES,a.tolist(),strict=True)))
        for k,a,b in zip(ARMS,mae,mse,strict=True)})


def _pairs(data, target, positive, safe):
    p, s = (np.flatnonzero(v) for v in (positive,safe))
    hits = excursion.disjoint_pairs(data['match'][p],data['match'][s],data['shot'][p],
                                    data['shot'][s],excursion.CALIPERS)
    i,j = p[hits[:,0]],s[hits[:,1]]
    names = (*ARMS,'observed_future','fixed_risk')
    by_group = np.concatenate((data['prediction'][:,:,-9:].mean(2),target[:,-9:].mean(1)[:,None]),1)
    continuous = np.c_[by_group.mean(2),data['match'][:,0]]
    dif = continuous[i]-continuous[j]
    return dict(pairs=len(i),positive_support=len(np.unique(data['shot'][p])),
        safe_support=len(np.unique(data['shot'][s])),
        median_context_distance=float(np.median(np.linalg.norm(
            (data['match'][i]-data['match'][j])/excursion.CALIPERS,axis=1))) if len(i) else None,
        arms={k:dict(ordered=int((a>0.).sum()), ties=int((a==0.).sum()),
                     median_difference=float(np.median(a)) if len(i) else None)
              for k,a in zip(names,dif.T,strict=True)},
        families={k:{a:int(n) for a,n in zip(names[:-1],h,strict=True)} for k,h in zip(
            excursion.FAMILIES,(by_group[i]>by_group[j]).sum(0).T,strict=True)},
        shots=np.c_[data['shot'][i],data['shot'][j]].tolist())


def score(out):
    """Outcomes are used only to stratify already sealed diagnostic forecasts."""
    p, seal = [read(out/k) for k in ('protocol.json','seal.json')]
    if any(sha(out/k)!=v for k,v in seal['assets'].items()) or sha(out/'executed_probe.py')!=p['producer_sha256']:
        raise ValueError('forecast evidence changed')
    lock = lag.load_inputs()
    ref = Path(p['reference'])
    if p['order']!=lock['ids'] or sha(ref/'seal.json')!=p['reference_seal']:
        raise ValueError('forecast evidence has a different cohort or reference')
    refp, refs = [read(ref/k) for k in ('protocol.json','seal.json')]
    with np.load(out/'decoders.npz',allow_pickle=False) as f:
        center,scale = f['center'],f['scale']
        np.testing.assert_array_equal(f['fit_ids'],np.sort(lock['ids'][:lag.STARTUP]))
    with np.load(out/'forecasts.npz',allow_pickle=False) as f:
        saved = dict(f)
    take, target, all_truth = [], [], []
    for rank,sid in enumerate(lock['ids'],1):
        d, down = io23.read_outcome('EAST',sid)
        all_truth.append(d)
        if rank<=lag.STARTUP:
            continue
        a = np.flatnonzero(saved['rank']==rank)
        if not len(a) or not (saved['shot'][a]==sid).all():
            raise ValueError('missing or misordered forecast discharge')
        raw_path = io23.path_for('EAST',sid)
        old = ref/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(raw_path)!=lock['hashes'][str(sid)] or sha(old)!=refs['curve_sha256'][old.name]:
            raise ValueError('raw clock or sealed parent changed during scoring')
        raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        signal = lag.prepare_actual(raw,rank,lock['mapping'])
        z,m,_,_ = shot_coordinates(signal,center,scale)
        view = {k:saved[k][a] for k in ('issue','ref','tail')}
        use,y = realized(z,m,view)
        e = view['issue'][use]
        t = raw['times']
        with np.load(old,allow_pickle=False) as f:
            np.testing.assert_array_equal(t,f['times'])
            hit = np.flatnonzero((f['fixed']>=refp['calibration']['0.2']) &
                    ((down-t>=-1e-9) if d else True))
        lead = np.round((down-t[e])*1000.,9) if d else np.zeros(len(e))
        release_lead = np.round((down-t[e+H+1])*1000.,9) if d else np.zeros(len(e))
        keep = (release_lead>=0.) if d else np.ones(len(e),bool)
        late = bool(d and (not len(hit) or round(float((down-t[hit[0]])*1000.),9)<=30.))
        target.append(y[keep])
        i = a[use][keep]
        take.append(dict(**{k:v[i] for k,v in saved.items() if k not in ('ref','tail')},
            disrupt=np.full(len(i),d), late=np.full(len(i),late), lead=lead[keep],
            release_lead=release_lead[keep], before_fp=(e[keep]+H+1<=hit[0]) if not d and len(hit)
            else np.full(len(i),not d)))
    if len(all_truth)!=360 or sum(all_truth)!=180 or sum(all_truth[:lag.STARTUP])!=8:
        raise ValueError('unexpected raw HDF5 outcome counts')
    x = excursion._stack(take)
    y = np.concatenate(target)
    d, safe = x['disrupt'], ~x['disrupt']
    near = d & (x['lead']<=125.) & (x['lead']>30.)
    actionable = near & (x['release_lead']>30.)
    masks = dict(suffix=np.ones(len(d),bool), safe=safe, near=near, before_deadline=actionable,
                 static_late_before_deadline=actionable & x['late'])
    metrics = {k:_losses(y[v],x['prediction'][v],x['rank'][v]) for k,v in masks.items()}
    matching = dict(near_vs_safe=_pairs(x,y,near,safe),
        eligible_vs_safe_before_fp=_pairs(x,y,actionable,safe & x['before_fp']),
        late_eligible_vs_safe_before_fp=_pairs(x,y,actionable & x['late'],safe & x['before_fp']))
    payload = dict(status='COMPLETED_DIAGNOSTIC_RESPONSE_SHADOW_NOT_ALARMS',total_shots=360,
        fit_shots=lag.STARTUP,scored_disruptive=172,scored_safe=168,scored_issues=len(y),
        sealed_issues=len(saved['issue']),targets_have_no_outcome_labels=True,
        losses=metrics,matched=matching,reference_sha256=p['reference_seal'])
    write(out/'result.json',payload)
    print({k:{'rows':v['rows'],'shots':v['shots'],
              'mae':{a:round(s['mae'],5) for a,s in v['arms'].items()}} for k,v in metrics.items()},flush=True)
    print({k:{'pairs':v['pairs'],'ordered':{a:s['ordered'] for a,s in v['arms'].items()}}
           for k,v in matching.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('predict','score'))
    parser.add_argument('output',type=Path)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    a=parser.parse_args()
    predict(a.reference,a.output) if a.stage=='predict' else score(a.output)
