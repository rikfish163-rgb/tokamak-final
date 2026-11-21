"""Test whether past-only diagnostic error scales improve shot-local representation updates."""

import argparse
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_write_delay_20261112 as delay
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


group, lag, layer = delay.group, delay.lag, delay.layer
ARCHIVE = io23.ROOT/'probe_outputs/commissioning_write_delay_east360_20261112_v2'
WAIT = 5
PRIOR_WINDOWS, PRECISION_GAIN = 4, 3.
RULES = ('unweighted', 'cold_full', 'local_diag', 'local_full')
NEW = ('current_5', 'earlier_5', *(f'{rule}_{age}' for rule in RULES[1:] for age in ('now', 'old')))
ARMS = (*group.OLD, *NEW)
CASES = (70320, 73405, 79649, 71076, 67900)


def sample_covariance(history, observed):
    """Estimate diagnostic covariance at the same forecast bin across distinct windows."""
    n = observed.sum(dim=1)
    count = n.clamp_min(1)[:, None, None, None]
    values = history.masked_fill(~observed[..., None, None], 0.)
    centered = (values - values.sum(dim=1, keepdim=True)/count).masked_fill(
        ~observed[..., None, None], 0.)
    covariance = torch.einsum('nkbj,nkbi->nij', centered, centered)
    return n, covariance/((n-1).clamp_min(1)*history.shape[2])[:, None, None]


def past_covariance(q, force, eligible, i):
    """Take only eligible forecasts issued recently and already mature at q[i]."""
    index = np.searchsorted(q+lag.H+1, q[i], side='right')[:, None]
    index = index - np.arange(group.bias.HISTORY, 0, -1)[None]
    clipped = index.clip(min=0)
    legal = ((index >= 0) & eligible[clipped] &
             (q[clipped] >= q[i, None]-group.bias.HISTORY*lag.STRIDE))
    return sample_covariance(force[clipped], torch.as_tensor(legal, device=force.device))


def apply_precision(force, cov, *, full):
    """Change direction using bounded shrinkage of the clipped forecast errors."""
    diag = cov.diagonal(dim1=-2, dim2=-1)
    if not full:
        return force/(1.+PRECISION_GAIN*diag[..., None, :])
    shrunk = .5*(cov + torch.diag_embed(diag))
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    return torch.linalg.solve(eye+PRECISION_GAIN*shrunk, force.transpose(-1, -2)).transpose(-1, -2)


def forces(now, old, cold, local, n):
    mixture = (PRIOR_WINDOWS*cold + n[:, None, None]*local)/(PRIOR_WINDOWS+n[:, None, None])
    return (now, old,
            *(apply_precision(x, cold, full=True) for x in (now, old)),
            *(apply_precision(x, mixture, full=full) for full in (False, True) for x in (now, old)))


def signed_residual(wave, observed, center, spread):
    pred = group.signed_levels(wave, center, spread)
    force = ((group.signed_levels(observed, center, spread) - pred.detach())/group.BETA).clamp(-1., 1.)
    return pred, force


@torch.no_grad()
def fit_cold(lock, args, decoder, path, protocol):
    """Replay completed calibration inputs in the frozen post-cold coordinates; no outcomes."""
    covariances, support = [], {}
    for sid in protocol['order'][:protocol['startup']]:
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        signal = lag.prepare_actual(raw, protocol['startup']+1, lock['mapping'])
        # The frozen parent itself defines a causal gradient-based input path.
        with torch.enable_grad():
            q, _, z, _, prior, phase, correction = group.parent_track(signal, args, decoder, path)
        eligible, fields, truth = group.feedback_waveform(signal, q, args[2], z.device)
        entries = []
        positions = np.flatnonzero(eligible)
        for j in range(0, len(positions), group.BATCH):
            i = positions[j:j+group.BATCH]
            _, _, _, wave, center, spread = group.signed_window(
                args[0], signal, q, i, phase, correction, prior, fields, decoder)
            entries.append(signed_residual(wave, truth[i], center, spread)[1])
        support[str(sid)] = len(positions)
        if len(positions) >= 2:
            eligible = torch.ones((1, len(positions)), dtype=torch.bool, device=z.device)
            covariances.append(sample_covariance(torch.cat(entries)[None], eligible)[1][0])
    if not covariances:
        raise ValueError('the cold covariance needs at least one shot with two mature forecasts')
    return torch.stack(covariances).mean(0), support


def estimate(signal, args, decoder, path, teacher, cold):
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path)
    eligible, fields, truth = group.feedback_waveform(signal, q, args[2], z.device)
    base = args[0]
    edges = base.cfg.window//base.cfg.patch-1
    states = prior.new_zeros((len(NEW), len(q)+1, edges, len(group.TRUSTED)))
    force = prior.new_zeros((len(q), lag.H//group.BIN, len(group.USE)))
    _, _, _, common = group.past_force_parts(q, force, eligible)
    count = np.zeros(len(q), dtype=np.int64)
    cos = np.full((len(NEW)-2, len(q)), np.nan, dtype=np.float32)
    positions = np.flatnonzero(eligible)
    for j in range(0, len(positions), group.BATCH):
        i = positions[j:j+group.BATCH]
        issued, current, state, wave, center, spread = group.signed_window(
            base, signal, q, i, phase, correction, prior, fields, decoder)
        pred, f = signed_residual(wave, truth[i], center, spread)
        force[i] = f
        _, now, old, _ = group.past_force_parts(q, force, eligible)
        n, cov = past_covariance(q, force, eligible, i)
        use = torch.as_tensor(common[i], device=z.device)
        choices = forces(now[i], old[i], cold, cov, n)
        g = group.signed_directions(issued, current, state, tuple(
            (pred, group.shape_force(x, teacher)*use[:, None, None]) for x in choices))
        states[:, i+1] = g
        count[i] = n.cpu().numpy()
        similarity = F.cosine_similarity(g[2:].flatten(2), g[:2].repeat((len(RULES)-1, 1, 1, 1)).flatten(2),
                                         dim=-1, eps=1e-8).cpu().numpy()
        cos[:, i[common[i]]] = similarity[:, common[i]]
    if not torch.isfinite(states).all():
        raise FloatingPointError('nonfinite weighted diagnostic direction')
    return q, h, z, beta, phase, correction, states, common, count, cos


@torch.no_grad()
def render(base, signal, track, parent, dose):
    q, h, z, beta, phase, correction, states, common, *_ = track
    release = q+lag.H+1
    slots = lag.delayed_slots(q, np.arange(len(z)))[0]
    active = np.flatnonzero(dose>0.)
    curves = np.broadcast_to(parent, (len(NEW), len(parent))).copy()
    parent_diff = dose_diff = 0.
    for offset in range(0, len(active), group.BATCH):
        i = active[offset:offset+group.BATCH]
        delayed = np.searchsorted(release+WAIT, i, side='right')
        if (delayed == 0).any() or not common[delayed-1].all():
            raise ValueError('a nonzero dose must use a mature eligible feedback')
        frac = group.handoff_fraction(release[delayed-1]+WAIT, i, base.cfg.patch)
        x, m = lag._windows(base, signal, i)
        view, _, err = group.forward_hidden(base, x, m, layer.patcher(phase[slots[i]], correction[slots[i]]),
            writes=states[:, delayed],
            handoff=(states[:, delayed-1], torch.as_tensor(frac, device=z.device, dtype=z.dtype)),
            target_norm=torch.as_tensor(dose[i], device=z.device, dtype=z.dtype))
        scores = (z[i]+((view-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
        curves[:, i] = scores[1:]
        parent_diff = max(parent_diff, float(np.abs(scores[0]-parent[i]).max()))
        dose_diff = max(dose_diff, float(err))
    if parent_diff>3e-6 or dose_diff>1e-5 or not np.isfinite(curves).all():
        raise ValueError(f'invalid parent or realized dosage: {parent_diff:g}/{dose_diff:g}')
    return curves, parent_diff, dose_diff


def seal_run(out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    ref, _, args, decoder, path = group.load_reference(lock)
    old, seal = (read(ARCHIVE/f'{k}.json') for k in ('protocol', 'seal'))
    same = ('order', 'startup', 'original', 'calibration', 'source_sha256')
    teach = delay.HISTORY/'teaching.json'
    if (any(ref[k]!=old[k] for k in same) or len(seal['curve_sha256'])!=360 or WAIT not in delay.WAITS
            or old['history_seal']!=sha(delay.HISTORY/'seal.json')
            or old['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or read(delay.HISTORY/'protocol.json')['teaching_sha256']!=sha(teach)):
        raise ValueError('the complete archived controls and frozen teacher must match the active protocol')
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(Path(__file__), out/'executed.py')
    base = args[0]
    frozen = [state_digest(m) for m in args[:2]]
    teacher = torch.as_tensor(read(teach)['weights'], device=base.center.device, dtype=torch.float32)
    thresholds = [*old['calibration'].values(), old['original']]
    files, witnesses = {}, {}
    parent_diff = dose_diff = archived_diff = prefix_diff = 0.
    bands = (21, 120, 360)
    protocol = dict(scope='EAST360_PAST_ONLY_FEEDBACK_PRECISION_DEVELOPMENT', **{k:old[k] for k in same},
        arms=ARMS, archive=str(ARCHIVE), archive_seal=sha(ARCHIVE/'seal.json'),
        reference_seal=old['reference_seal'], prior_windows=PRIOR_WINDOWS, precision_gain=PRECISION_GAIN,
        fixed_correlation_shrinkage=.5, same_shot_history_steps=group.bias.HISTORY,
        diagnostic_channels=[io23.RAW23[lag.CHANNELS[k]] for k in group.USE],
        feedback='issue direct 25-point forecast, clip signed errors; use earlier mature forces from the same shot '
                 'at issue time for covariance; never use the evaluated error for its own covariance',
        basis='same original group teacher, issue/release Jacobians, edges, dose, +5 wait and 8-point handoff',
        scoring='freeze all curves before reading HDF5 outcomes; no new label access or readout training')
    write(out/'protocol.json', protocol)
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcomes forbidden during prediction')):
        cold, support = fit_cold(lock, args, decoder, path, protocol)
        write(out/'cold_precision.json', dict(per_shot_windows=support, covariance=cold.cpu().tolist()))
        print(dict(stage='cold_covariance_sealed', windows=sum(support.values()), seconds=time.monotonic()-started),
              flush=True)
        for rank, sid in enumerate(old['order'], 1):
            name = f'{rank:04d}_{sid}.npz'
            source = ARCHIVE/'shots'/name
            if (sha(source)!=seal['curve_sha256'][name]
                    or (rank<=old['startup'] and sha(io23.path_for('EAST', sid))!=lock['hashes'][str(sid)])):
                raise ValueError('changed archived curve or cold inputs')
            with np.load(source, allow_pickle=False) as f:
                fixed, parent, release, eligible = (f[k] for k in ('fixed', 'bias_only', 'release', 'eligible'))
                dose = f['dose'][delay.WAITS.index(WAIT)]
                saved = np.stack([f[k] for k in NEW[:2]])
            if rank<=old['startup']:
                curves = np.broadcast_to(parent, (len(NEW), len(parent)))
                count = np.zeros(len(release), dtype=np.int64)
                cos = np.full((len(NEW)-2, len(release)), np.nan, dtype=np.float32)
            else:
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                if sha(io23.path_for('EAST', sid))!=lock['hashes'][str(sid)] or len(raw['times'])!=len(parent):
                    raise ValueError('changed input or sample count')
                signal = lag.prepare_actual(raw, rank, lock['mapping'])
                track = estimate(signal, args, decoder, path, teacher, cold)
                count, cos = track[-2:]
                np.testing.assert_array_equal(track[0]+lag.H+1, release)
                np.testing.assert_array_equal(track[-3], eligible)
                curves, pe, de = render(base, signal, track, parent, dose)
                parent_diff, dose_diff = max(parent_diff, pe), max(dose_diff, de)
                archived_diff = max(archived_diff, float(np.abs(curves[:2]-saved).max()))
                np.testing.assert_allclose(curves[:2], saved, atol=3e-6, rtol=0.)
                for t in thresholds:
                    np.testing.assert_array_equal(curves[:2]>=t, saved>=t)
                if rank in bands or sid in CASES:
                    if rank in bands:
                        end = min(550, len(parent))
                    else:
                        alarm = np.flatnonzero((curves[-2]>=old['calibration']['0.2']) & (dose>0.))
                        end = min(int(alarm[0])+2 if len(alarm) else 550, len(parent))
                    prefix = {k:signal[k][:end] for k in ('values', 'available')}
                    partial = estimate(prefix, args, decoder, path, teacher, cold)
                    np.testing.assert_array_equal(partial[-2], count[release<end])
                    short, *_ = render(base, prefix, partial, parent[:end], dose[:end])
                    np.testing.assert_allclose(short, curves[:, :end], atol=3e-6, rtol=0.)
                    for t in thresholds:
                        np.testing.assert_array_equal(short>=t, curves[:, :end]>=t)
                    prefix_diff = max(prefix_diff, float(np.abs(short-curves[:, :end]).max()))
                    witnesses[str(sid)] = end
            dest = out/'shots'/name
            with dest.open('xb') as stream:
                np.savez_compressed(stream, fixed=fixed, bias_only=parent, release=release, eligible=eligible,
                    dose=dose, covariance_windows=count, gradient_cosine=cos, **dict(zip(NEW, curves, strict=True)))
            files[name] = sha(dest)
            if rank in (20, 21) or rank%60==0:
                print(dict(stage='sealed_precision_curves', rank=rank, seconds=time.monotonic()-started), flush=True)
    if frozen!=[state_digest(m) for m in args[:2]]:
        raise ValueError('frozen upstream model changed')
    write(out/'seal.json', dict(curves_sha256=files, prefix_witnesses=witnesses,
        max_parent_diff=parent_diff, max_archived_diff=archived_diff, max_prefix_diff=prefix_diff,
        max_realized_dose_error=dose_diff, elapsed_s=time.monotonic()-started))


def score(out):
    p, seal = (read(out/f'{k}.json') for k in ('protocol', 'seal'))
    if len(seal['curves_sha256'])!=360 or p['archive_seal']!=sha(ARCHIVE/'seal.json'):
        raise ValueError('whole fixed 360 must be sealed before scoring')
    archived = read(ARCHIVE/'per_shot.json')
    per = {b:{k:[] for k in ARMS} for b in p['calibration']}
    counts = np.zeros(group.bias.HISTORY+1, dtype=np.int64)
    cosine = defaultdict(list)
    changed = 0
    confirms = {band:{k:np.zeros((3, 2), dtype=int) for k in ARMS} for band in p['calibration']}
    cases = []
    for rank, sid in enumerate(p['order'], 1):
        name = f'{rank:04d}_{sid}.npz'
        dest = out/'shots'/name
        if sha(dest)!=seal['curves_sha256'][name]:
            raise ValueError('sealed scores changed')
        y, down, t = group.hdf._raw_truth(sid)
        with np.load(dest, allow_pickle=False) as r:
            release, eligible, dose = (r[k] for k in ('release', 'eligible', 'dose'))
            curves = {k:r[k] for k in ARMS}
            support = r['covariance_windows']
            activity = dose>0.
            changed += int(activity.sum())
            ready = np.searchsorted(release+WAIT, np.arange(len(t)), side='right')
            if not np.r_[False, eligible][ready[activity]].all():
                raise ValueError('unavailable writes in the sealed stream')
            counts += np.bincount(support[eligible], minlength=len(counts))
            for k, c in zip(NEW[2:], r['gradient_cosine'], strict=True):
                cosine[k].extend(c[eligible].tolist())
            for k in NEW:
                np.testing.assert_array_equal(curves[k][~activity], curves['bias_only'][~activity])
            for band, post in p['calibration'].items():
                tau = p['original'] if rank<=p['startup'] else post
                for k, v in curves.items():
                    result = dict(shot=sid, disruptive=y, down_s=down, **score_curve(t, v, y, down, tau))
                    if k in archived[band] and result!=archived[band][k][rank-1]:
                        raise ValueError('archived control first alarm changed')
                    per[band][k].append(result)
                    crossings = v>=tau
                    for j, n in enumerate((2, 3, 5)):
                        valid = np.lib.stride_tricks.sliding_window_view(crossings, n).all(1) if len(t)>=n else []
                        first = np.flatnonzero(valid)
                        if len(first):
                            timely = y and round(float((down-t[first[0]+n-1])*1000), 9)>30.
                            confirms[band][k][j] += (timely, not y)
            if sid in CASES:
                arms = {k:per['0.2'][k][-1] for k in ARMS}
                tau = p['calibration']['0.2']
                evidence = {}
                for k in NEW:
                    visits = np.flatnonzero(curves[k]>=tau)
                    if len(visits) and (not y or t[visits[0]]<=down):
                        i = int(visits[0])
                        g = ready[i]-1 if activity[i] else -1
                        evidence[k] = dict(index=i, seconds=float(t[i]), dose=float(dose[i]),
                            score=float(curves[k][i]), parent=float(curves['bias_only'][i]),
                            covariance_windows=int(support[g]) if g>=0 else None)
                cases.append(dict(shot=sid, rank=rank, scores=arms, evidence=evidence))
    metrics = {b:{k:lead.summarize(v) for k,v in rows.items()} for b,rows in per.items()}
    pairs = [('bias_only', k) for k in NEW]
    pairs += [('current_5', k) for k in NEW[2::2]]
    pairs += [(NEW[2*j+1], NEW[2*j]) for j in range(len(RULES))]
    paired = {b:{f'{a}__to__{k}':dict(flips=flips(rows[a], rows[k]), lead=lead.paired(rows[a], rows[k]))
                 for a,k in pairs} for b,rows in per.items()}
    age_interactions = {}
    for b, data in metrics.items():
        effects = {rule:data[NEW[2*j]]['by_deadline']['30']['tp']-data[NEW[2*j+1]]['by_deadline']['30']['tp']
                   for j,rule in enumerate(RULES)}
        age_interactions[b] = dict(latest_minus_earlier=effects,
            local_full_vs_fixed_full=effects['local_full']-effects['cold_full'])
    write(out/'per_shot.json', per)
    write(out/'result.json', dict(raw_verified=360*len(ARMS)*len(p['calibration']), metrics=metrics,
        paired=paired, age_interactions=age_interactions, write_endpoints=changed,
        covariance_windows_histogram=counts.tolist(),
        gradient_cosine_q10_median_q90={k:np.quantile(v, [.1,.5,.9]).tolist() for k,v in cosine.items()},
        confirmation_lengths=(2, 3, 5), confirmation_counts={b:{k:v.tolist() for k,v in r.items()}
                     for b,r in confirms.items()}, named_cases=cases))
    print({b:{k:(v['by_deadline']['30']['tp'], v['fp']) for k,v in r.items()} for b,r in metrics.items()}, flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    out = parser.parse_args().out.resolve()
    seal_run(out)
    score(out)
