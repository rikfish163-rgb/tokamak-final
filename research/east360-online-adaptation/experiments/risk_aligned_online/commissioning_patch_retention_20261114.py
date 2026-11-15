"""Test recoverability of measured 8-sample waveforms in the frozen source encoder."""

import argparse
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch

from . import commissioning_weak_signal_probe_20261114 as probe
from . import io23, simple_data
from .commissioning_causal_reference_20261114 import PAIRED
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .operating_point_audit import best_at_count


COLS = tuple(probe.COLUMNS)
RADIATION = tuple(COLS.index(io23.RAW23.index(k)) for k in
                  ('sxr_skew', 'sxr_var', 'xuv_skew', 'xuv_var'))
PATCH = 8
SLOTS = probe.WIDTH//PATCH
BASIS = np.sqrt(2./PATCH)*np.cos(np.pi*np.arange(PATCH)[:, None]*(np.arange(PATCH)+.5)/PATCH)
BASIS[0] = 1./np.sqrt(PATCH)
STAGES = ('embedding', 'block1', 'block2')
WAVES = ('raw_exact', 'raw_dc', *(k+'_decoded' for k in STAGES))
ARMS = (*WAVES, 'raw_dc_refit', 'block2_direct', 'cls_direct', 'strong_parent')
ERRORS = ('dc', 'within_patch', 'radiation_dc', 'radiation_within_patch')


def temporal_coordinates(x):
    return np.einsum('...tc,kt->...kc', x, BASIS)


def dc_only(wave):
    x = np.zeros_like(wave)
    x[..., 0, :] = wave[..., 0, :]
    return x


def fit_decoder(x, truth, mass, ridge=.02):
    """One patch decoder per layer, with a slot intercept but no shot identity."""
    x, truth = np.asarray(x, np.float64), np.asarray(truth, np.float64)
    xmean, tmean = (np.einsum('n,npd->pd', mass, a) for a in (x, truth))
    centered = x-xmean
    scale = np.sqrt(np.maximum(np.einsum('n,npd->d', mass, centered**2)/x.shape[1], .05**2))
    w = np.repeat(np.sqrt(mass/x.shape[1]), x.shape[1])[:, None]
    a = (centered/scale).reshape(-1, x.shape[-1])*w
    b = (truth-tmean).reshape(-1, truth.shape[-1])*w
    coef = np.linalg.solve(a.T@a+ridge*np.eye(a.shape[1]), a.T@b)/scale[:, None]
    return dict(coef=coef, bias=tmean-xmean@coef)


def decode(x, model):
    return x@model['coef']+model['bias']


def recovery_error(estimate, truth):
    e = np.abs(estimate-truth)
    slices = (e[:, :, :1], e[:, :, 1:], e[:, :, :1, RADIATION], e[:, :, 1:, RADIATION])
    return np.column_stack([v.mean(axis=tuple(range(1,v.ndim))) for v in slices])


@torch.no_grad()
def encoder_views(base, signal, ends, transform_for=None):
    """Capture time-token outputs before the final normalization and CLS readout."""
    if (base.cfg.window, base.cfg.patch, base.cfg.dim, base.cfg.layers, base.cfg.input_view) != (
            probe.WIDTH, PATCH, 64, 2, 'source'):
        raise ValueError('the qualified 128/8/64/two-block source encoder is required')
    op, good = probe.operating(signal, ends)
    widths = dict(raw=(SLOTS,PATCH,len(COLS)), embedding=(SLOTS,64),
                  block1=(SLOTS,64), block2=(SLOTS,64), cls=(64,), base=(1,), source=(1,))
    data = {k:np.empty((len(ends),*v), np.float32) for k,v in widths.items()}
    for j in range(0,len(ends),probe.BATCH):
        sl = slice(j,j+probe.BATCH)
        t = ends[sl]
        seen = {}
        x,m = probe.lag._windows(base,signal,t)
        known = (m & x.isfinite())[..., list(COLS)]
        z = ((torch.where(known,x[...,list(COLS)],base.center[list(COLS)])
              -base.center[list(COLS)])/base.scale[list(COLS)]).asinh()
        good[sl] &= known.all(dim=(1,2)).cpu().numpy()
        data['raw'][sl] = z.reshape(len(t),SLOTS,PATCH,len(COLS)).cpu().numpy()

        def hook(k, time_tokens):
            def capture(_module, args, output=None):
                a = args[0] if output is None else output
                seen[k] = (a[:,:SLOTS] if time_tokens else a).detach()
            return capture

        handles = [base.embedding.register_forward_hook(hook('embedding', True)),
                   base.encoder.layers[0].register_forward_hook(hook('block1', True)),
                   base.encoder.layers[1].register_forward_hook(hook('block2', True)),
                   base.risk_head.register_forward_pre_hook(hook('cls', False))]
        if transform_for is not None:
            transform, score = transform_for(t)
            handles.append(base.embedding.register_forward_pre_hook(lambda _m,a: (transform(a[0]),)))
        try:
            source_logit = base(x,m)['logit']
        finally:
            for h in handles:
                h.remove()
        data['source'][sl,0] = source_logit.sigmoid().cpu().numpy()
        data['base'][sl,0] = data['source'][sl,0] if transform_for is None else score(seen['cls'])
        for k,v in seen.items():
            data[k][sl] = v.cpu().numpy()
    return dict(**data,op=op,supported=good)


def flatten(x):
    return x.reshape(len(x), int(np.prod(x.shape[1:])))


def source_pool(base, ids):
    rows = []
    adv = io23.advanced_times()
    for sid in ids:
        sig, ends, outcome = probe.source_training_points(sid, adv)
        if not len(ends):
            continue
        view = encoder_views(base, sig, ends)
        good = view.pop('supported')
        if good.any():
            rows.append(dict(**{k:v[good] for k,v in view.items()},
                             sid=np.full(good.sum(),sid), y=np.full(good.sum(),outcome,bool)))
    return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def fit(base):
    ids = simple_data.source_roles()['train']
    if len(ids)!=1353 or len(set(ids))!=1353:
        raise ValueError('fixed TRAIN1353 required')
    s = source_pool(base,ids)
    actual = temporal_coordinates(s['raw'])
    t = actual.reshape(len(actual),SLOTS,-1)
    x,y,shot = s['op'],s['y'],s['sid']
    mass = probe.training_weights(shot,y)
    decoders = {k:fit_decoder(s[k],t,mass) for k in STAGES}
    heads = {k:probe.fit_probe(x,v,shot,y) for k,v in (
        ('raw_exact',flatten(actual)), ('raw_dc_refit',flatten(actual[:,:,:1])),
        ('block2_direct',flatten(s['block2'])), ('cls_direct',s['cls']))}
    return decoders,heads,probe.train_counts(shot,y)


def packet(view, decoders, heads):
    actual = temporal_coordinates(view['raw'])
    shape = actual.shape
    restored = [decode(view[k],decoders[k]).reshape(shape) for k in STAGES]
    dc = dc_only(actual)
    risk = [probe.apply_probe(view['op'],flatten(a),heads['raw_exact'])
            for a in (actual,dc,*restored)]
    risk.extend(probe.apply_probe(view['op'],x,heads[k]) for k,x in (
        ('raw_dc_refit',flatten(actual[:,:,:1])), ('block2_direct',flatten(view['block2'])),
        ('cls_direct',view['cls'])))
    risk.append(view['base'][:,0])
    err = np.stack([recovery_error(a,actual) for a in (dc,*restored)],axis=1)
    return np.column_stack(risk).astype(np.float32),err.astype(np.float32)


def reconstruction_stats(shot_mean_errors):
    x = np.stack(shot_mean_errors).mean(0)
    return {k+'_'+j:float(d) for k,row in zip(WAVES[1:],x,strict=True)
            for j,d in zip(ERRORS,row,strict=True)}


def source_hold(base, decoders, heads):
    held = source_pool(base,simple_data.source_roles()['val'])
    score, err = packet(held,decoders,heads)
    ids = np.unique(held['sid'])
    per = [held['sid']==sid for sid in ids]
    peaks = np.stack([score[j].max(0) for j in per])
    y = np.array([held['y'][j][0] for j in per])
    n = int(y.sum())
    auc = (rankdata(peaks,axis=0)[y].sum(0)-n*(n+1)/2)/(n*(len(y)-n))
    return dict(support=probe.train_counts(held['sid'],held['y']),
                peak_auc=dict(zip(ARMS,auc.tolist(),strict=True)),
                recovery_mae=reconstruction_stats([err[j].mean(0) for j in per]))


def projection_geometry(base):
    # Masks fixed: the full value map is linear from all 8*23 coordinates.
    w = base.embedding.weight.detach().cpu().numpy().reshape(64,PATCH,46)
    rows = w[:,:,:23].reshape(64,-1).astype(np.float64)
    _,s,v = np.linalg.svd(rows,full_matrices=False)
    keep = s > s[0]*1e-10
    coords = np.einsum('ft,cq->fctq',BASIS,np.eye(23)[list(COLS)]).reshape(PATCH,len(COLS),-1)
    fraction = np.square(coords@v[keep].T).sum(-1)
    isolated = np.linalg.svd(w[:,:,list(COLS)].reshape(64,-1),compute_uv=False)
    return dict(full_value_rank=int(keep.sum()),isolated_selected_singular_values=isolated.tolist(),
                recoverable_linear_energy=fraction.tolist(),
                columns=[io23.RAW23[k] for k in COLS],
                meaning='row-space projection of a unit DCT coordinate with other value lanes unconstrained')


def transform_schedule(signal,args,decoder,path):
    q,h,z,beta,_,phase,correction = probe.group.parent_track(signal,args,decoder,path)

    def at(ends):
        issued = probe.lag.delayed_slots(q,ends)[0]
        adjust = probe.layer.patcher(phase[issued],correction[issued])
        def parent(cls):
            return (z[ends]+((cls-h[ends])*beta).sum(-1)).sigmoid().cpu().numpy()
        return adjust,parent

    return at


def observe(out):
    begun = monotonic()
    lock,p,seal,args,decoder,path = probe.history.inputs()
    base = args[0]
    old_state = {k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    decoders,heads,counts = fit(base)
    hold = source_hold(base,decoders,heads)
    write(out/'source.json',dict(train=counts,held=hold,heads=heads,projection=projection_geometry(base),
          decoders={k:{a:b.tolist() for a,b in v.items()} for k,v in decoders.items()}))
    print(dict(stage='source_frozen',held=hold['peak_auc'],seconds=round(monotonic()-begun,2)),flush=True)
    names, witnesses, parity = [], {}, 0.
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('EAST outcome not observable')),
          patch.object(probe.group.hdf,'_raw_truth',side_effect=RuntimeError('EAST outcome not observable'))):
        for rank,sid in enumerate(p['order'],1):
            if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError('locked EAST input changed')
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            ends = probe.end_grid(len(raw['times']))
            signal = probe.lag.prepare_actual(raw,rank,lock['mapping'])
            arc = probe.history.archived_shot(rank,sid,seal)
            adapt = rank>p['startup']
            at = transform_schedule(signal,args,decoder,path) if adapt else None
            view = encoder_views(base,signal,ends,at)
            scores,errors = packet(view,decoders,heads)
            if adapt and len(ends):
                actual, expected = scores[:,-1],arc['bias_only'][ends]
                probe.check_thresholds(actual,expected,tuple(p['calibration'].values()))
                parity = max(parity,float(np.max(np.abs(actual-expected))))
            else:
                # Cold outcomes and original scores are never retested using probe views.
                scores[:,-1] = arc['bias_only'][ends]
            if rank in (21,120,360) or sid in (69813,73790,73405,79649):
                n = min(len(signal['values']),616)
                sub = {k:v[:n] for k,v in signal.items() if k in ('values','available')}
                sub_at = transform_schedule(sub,args,decoder,path) if adapt else None
                j = ends<n
                short = encoder_views(base,sub,ends[j],sub_at)
                a,b = packet(short,decoders,heads)
                np.testing.assert_allclose(a,scores[j],rtol=0.,atol=4e-5)
                np.testing.assert_allclose(b,errors[j],rtol=0.,atol=3e-5)
                np.testing.assert_array_equal(short['supported'],view['supported'][j])
                witnesses[str(sid)] = dict(points=n,frames=int(j.sum()),max_score_diff=float(
                    np.max(np.abs(a-scores[j]))))
            f = f'{rank:04d}_{sid}.npz'
            with (out/'shots'/f).open('xb') as stream:
                np.savez_compressed(stream,ends=ends,available=view['supported'],scores=scores,error=errors)
            names.append(f)
            if rank in (20,120,240,360):
                print(dict(stage='sealed',shots=rank,seconds=round(monotonic()-begun,2)),flush=True)
    for k,v in base.state_dict().items():
        np.testing.assert_array_equal(v.detach().cpu().numpy(),old_state[k].numpy())
    write(out/'seal.json',dict(arms=ARMS,completed=names,reference=str(probe.history.ARCHIVE),
          order=p['order'],original_threshold=p['original'],calibration=p['calibration'],
          cold=p['startup'],prefixes=witnesses,parent_parity_max=parity,
          clock='q=127+4k, observed samples up to and including q',
          score_scope='diagnostic only; no new alarms or target-outcome fit',seconds=monotonic()-begun))


def matched_report(comparisons, rows):
    result = {}
    for name, config in comparisons.items():
        report = result[name] = {}
        for group in ('no_first_alarm','late_first_alarm'):
            matches = [p for p in config['pairs'] if p['positive_group']==group]
            margins, usable = [], []
            for pair in matches:
                a,b = (rows[pair[k]] for k in ('positive_shot','safe_shot'))
                check = []
                for k in ('earlier_32','current'):
                    t = pair['anchors'][k]
                    i,j = (int(np.searchsorted(v['ends'],t[side+'_index']))
                           for v,side in ((a,'positive'),(b,'safe')))
                    if (a['ends'][i],b['ends'][j]) != (t['positive_index'],t['safe_index']):
                        raise ValueError('pair endpoint not on sealed clock')
                    if a['available'][i] and b['available'][j]:
                        # Other scores are logits; convert the parent probability for comparability.
                        delta = a['scores'][i].astype(np.float64)-b['scores'][j]
                        z = np.clip([a['scores'][i,-1],b['scores'][j,-1]],1e-6,1.-1e-6)
                        delta[-1] = np.log(z[0]/(1.-z[0]))-np.log(z[1]/(1.-z[1]))
                        check.append(delta)
                if len(check)==2:
                    margins.append(check)
                    usable.append(dict(positive=pair['positive_shot'],safe=pair['safe_shot'],
                                       margins=np.stack(check).tolist()))
            m = np.asarray(margins).reshape(-1,2,len(ARMS))
            positive = m>0.
            report[group] = dict(n=len(usable),matched=len(matches),
                earlier=positive[:,0].sum(0).tolist(),later=positive[:,1].sum(0).tolist(),
                both=positive.all(1).sum(0).tolist(),pair_rows=usable)
    return result


def score(out):
    p = read(out/'seal.json')
    ref = probe.history.ARCHIVE
    old, archive = (read(ref/f'{k}.json') for k in ('per_shot','seal'))
    ref_arms = ('bias_only','current_5')
    original = {k:old['0.2'][k] for k in ref_arms}
    names = [f'{rank:04d}_{sid}.npz' for rank,sid in enumerate(p['order'],1)]
    if (p['completed']!=names or len(names)!=360 or
            p['order']!=[r['shot'] for r in original['current_5']]):
        raise ValueError('a complete sealed 360 is required before inspecting outcomes')
    groups = ('all_timely','near_30_125ms')
    peaks = {g:{k:[] for k in ARMS} for g in groups}
    examples, hist, absent_safe = {}, [], []
    tally = {k:dict(timely=0,false=0) for k in ref_arms}
    for rank,(sid,name) in enumerate(zip(p['order'],names,strict=True),1):
        y,down,t = probe.group.hdf._raw_truth(sid)
        ref_curve = probe.history.archived_shot(rank,sid,archive)
        cold = rank<=p['cold']
        tau = p['original_threshold'] if cold else p['calibration']['0.2']
        for k in ref_arms:
            a = score_curve(t,ref_curve[k],y,down,tau)
            if dict(shot=sid,disruptive=y,down_s=down,**a)!=original[k][rank-1]:
                raise ValueError('archived first alarm differs from HDF5 recomputation')
            tally[k]['timely']+=int(a['timely'])
            tally[k]['false']+=int(not y and a['alarm'])
        with np.load(out/'shots'/name,allow_pickle=False) as stored:
            v = dict(stored)
        e, good = v['ends'], v['available']
        np.testing.assert_array_equal(e,probe.end_grid(len(t)))
        if cold:
            continue
        examples[sid] = v
        lead = np.round((down-t[e])*1e3,9) if y else np.full(len(e),np.inf)
        old_result = original['current_5'][rank-1]
        kind = ('timely' if old_result['timely'] else 'late' if old_result['alarm'] else 'never') if y else (
            'false' if old_result['alarm'] else 'safe')
        if not y and not good.any():
            absent_safe.append(sid)
            continue
        allowed = (good & (lead>30.), good & (lead>30.) & ((lead<=125.) | (not y)))
        for g,eligible in zip(groups,allowed,strict=True):
            peak = v['scores'][eligible].max(0) if eligible.any() else [None]*len(ARMS)
            for k, val in zip(ARMS,peak,strict=True):
                prob = (float(val) if k=='strong_parent' else float(expit(np.float64(val)))) if val is not None else None
                peaks[g][k].append(dict(shot=sid,disruptive=y,peak=prob))
        close = ((lead<=125.) & (lead>30.)) if y else (
            (t[e] <= t[np.flatnonzero(ref_curve['current_5']>=tau)[0]]) if kind=='false' else np.ones(len(e),bool))
        hist.append(dict(shot=sid,kind=kind,n=int(good.sum()),near_n=int((close & good).sum()),
                         whole_error=v['error'][good].mean(0).tolist() if good.any() else None,
                         near_error=v['error'][close & good].mean(0).tolist() if (close & good).any() else None))
    if absent_safe:
        raise ValueError(f'no common safety peaks for {absent_safe}')
    equal = {g:{str(n):{k:{m:v[m] for m in ('tp','fp')} for k,r in a.items()
                        for v in [best_at_count(r,n)]} for n in (12,21,30)} for g,a in peaks.items()}
    checks = matched_report(read(PAIRED)['comparisons'],examples)
    subsets = dict(suffix=('whole_error',lambda r:True),
                   never_near=('near_error',lambda r:r['kind']=='never'),
                   late_near=('near_error',lambda r:r['kind']=='late'),
                   false_before_alarm=('near_error',lambda r:r['kind']=='false'))
    recovered = {}
    for k,(field,include) in subsets.items():
        seen = [r for r in hist if include(r) and r[field] is not None]
        recovered[k] = dict(shots=len(seen),frames=sum(r['n' if field=='whole_error' else 'near_n'] for r in seen),
                            mae=reconstruction_stats([r[field] for r in seen]))
    write(out/'result.json',dict(arm_order=ARMS, full360_locked_reference=tally,
          recovery=recovered, after_seal_peak_ranking_suffix=equal, paired=checks, support=hist,
          qualification='developed stream; ranking only, no threshold deployment or target label training'))
    print(dict(reference=tally,peaks=equal,recovery=recovered),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--score',action='store_true')
    a = parser.parse_args()
    (score if a.score else observe)(a.output.resolve())
