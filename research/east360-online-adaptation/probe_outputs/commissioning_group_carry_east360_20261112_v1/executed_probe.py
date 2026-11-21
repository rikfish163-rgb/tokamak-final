"""Compare resetting and carrying first-block coefficients at the same EAST feedback clock and dose."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_group_token_20261112 as group
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


lag, layer = group.lag, group.layer
HISTORY = io23.ROOT/'probe_outputs/commissioning_signed_innovations_east360_20261112_v1'
RESET_CURRENT, RESET_EARLIER, CARRY_CURRENT, CARRY_EARLIER = NEW = (
    'reset_current', 'reset_earlier', 'carry_current', 'carry_earlier')
ARMS = (*group.OLD, *NEW)
PAIRS = ((group.bias.BIAS, RESET_CURRENT), (group.bias.BIAS, CARRY_CURRENT),
         (RESET_CURRENT, CARRY_CURRENT), (RESET_EARLIER, CARRY_EARLIER), (CARRY_EARLIER, CARRY_CURRENT))
RETENTION = .5


def carried_directions(steps, eligible):
    """Keep an exponentially decaying direction in the shot; ineligible slots cannot write."""
    unit = F.normalize(steps[:, 1:].flatten(-2), dim=-1, eps=group.EPS).reshape_as(steps[:, 1:])
    bank = torch.zeros_like(steps[:, 0])
    result = torch.zeros_like(steps)
    for j, available in enumerate(eligible, 1):
        bank = RETENTION*bank
        if available:
            bank = bank + (1.-RETENTION)*unit[:, j-1]
            result[:, j] = bank
    return result


@torch.no_grad()
def render(base, signal, q, h, z, beta, phase, correction, directions, parent, budget):
    curves = np.broadcast_to(parent, (len(directions), len(parent))).copy()
    ends = np.flatnonzero(budget>0.)
    parent_error = dose_error = 0.
    for start in range(0, len(ends), group.BATCH):
        i = ends[start:start+group.BATCH]
        slot, _ = lag.delayed_slots(q, i)
        frac = group.handoff_fraction(q[slot-1]+lag.H+1, i, base.cfg.patch)
        x, mask = lag._windows(base, signal, i)
        view, _, err = group.forward_hidden(
            base, x, mask, layer.patcher(phase[slot], correction[slot]), writes=directions[:, slot],
            handoff=(directions[:, slot-1], torch.as_tensor(frac, device=z.device, dtype=z.dtype)),
            target_norm=torch.as_tensor(budget[i], device=z.device, dtype=z.dtype))
        prob = (z[i]+((view-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
        curves[:, i] = prob[1:]
        parent_error = max(parent_error, float(np.max(np.abs(prob[0]-parent[i]))))
        dose_error = max(dose_error, float(err))
    if parent_error>3e-6 or dose_error>1e-5 or not np.isfinite(curves).all():
        raise ValueError(f'parent or equal-dose replay failed: {parent_error:g}/{dose_error:g}')
    return curves, len(ends), parent_error, dose_error


def predict_shot(signal, args, decoder, path, weights, parent, release, eligible, budget):
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path)
    base = args[0]
    available, fields, waveform = group.feedback_waveform(signal, q, args[2], z.device)
    states, common = group._estimate_innovation(
        base, signal, q, available, phase, correction, prior, fields, waveform, decoder, weights)
    np.testing.assert_array_equal(q+lag.H+1, release)
    np.testing.assert_array_equal(common, eligible)
    legal = np.r_[False, common][np.searchsorted(release, np.arange(len(parent)), side='right')]
    if np.any((budget>0.) & ~legal):
        raise ValueError('nonzero budget outside the original released feedback support')
    g = states[-2:]
    directions = torch.cat((g, carried_directions(g, common)))
    return render(base, signal, q, h, z, beta, phase, correction, directions, parent, budget)


def run(out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    p, ref, args, decoder, path = group.load_reference(lock)
    hist, old = (read(HISTORY/k) for k in ('protocol.json', 'seal.json'))
    frozen = ('order', 'startup', 'original', 'calibration', 'source_sha256')
    asset = HISTORY/'teaching.json'
    if (any(p[k]!=hist[k] for k in frozen)
            or hist['scope']!='EAST360_SAME_SHOT_FORCE_INNOVATION_DEVELOPMENT'
            or tuple(hist['arms'])!=(*group.OLD, *group.INNOVATION)
            or hist['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or hist['teaching_sha256']!=sha(asset) or len(old['curve_sha256'])!=360):
        raise ValueError('the original current/earlier innovation and dose archive is required')
    weights = torch.as_tensor(read(asset)['weights'], device=args[0].center.device, dtype=torch.float32)
    before = [state_digest(m) for m in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__), out/'executed_probe.py')
    copyfile(Path(group.__file__), out/'executed_group.py')
    write(out/'protocol.json', dict(scope='EAST360_SHOT_LOCAL_COEFFICIENT_CARRY_DEVELOPMENT',
        **{k:p[k] for k in frozen}, arms=ARMS, reference=str(group.REFERENCE),
        reference_seal=hist['reference_seal'], historical_reference=str(HISTORY),
        historical_seal=sha(HISTORY/'seal.json'), teaching_sha256=sha(asset),
        executed_sha256=sha(out/'executed_probe.py'), group_sha256=sha(out/'executed_group.py'),
        clock='unchanged issue, H25 feedback and q+26 release; no outcomes until all curves are sealed',
        state='unit step mixed with prior coefficient state; decay without writing at ineligible slots; reset per shot',
        retention=RETENTION, horizon=lag.H, stride=lag.STRIDE,
        control='current and previously released innovation, each reset and carried; same shadow forecast and Jacobian',
        writing='original realized L2 and zero-dose endpoints, same 8-point handoff, same first-block token exchange'))
    hashes, witnesses, maxima = {}, {}, np.zeros(3)
    endpoints = 0
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome forbidden before seal')):
        for rank, sid in enumerate(p['order'], 1):
            name = f'{rank:04d}_{sid}.npz'
            raw = io23.path_for('EAST', sid)
            a, b = (root/'shots'/name for root in (group.REFERENCE, HISTORY))
            if (sha(a)!=ref['curve_sha256'][name] or sha(b)!=old['curve_sha256'][name]
                    or sha(raw)!=lock['hashes'][str(sid)]):
                raise ValueError('fixed data or replay reference changed')
            with np.load(a, allow_pickle=False) as f, np.load(b, allow_pickle=False) as j:
                t, fixed, parent = (f[k] for k in ('times', *group.OLD))
                release, eligible, dose = (j[k] for k in ('release', 'eligible', 'dose'))
                archived = np.stack([j[k] for k in group.INNOVATION[-2:]])
                np.testing.assert_array_equal(release, f['release'])
            if rank<=p['startup']:
                if np.any(dose) or len(release):
                    raise ValueError('preparation segment has feedback writes')
                curves = np.broadcast_to(parent, (len(NEW), len(t)))
            else:
                record = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(record['times'], t)
                signal = lag.prepare_actual(record, rank, lock['mapping'])
                curves, n, e, d = predict_shot(signal, args, decoder, path, weights, parent, release, eligible, dose)
                endpoints += n
                np.testing.assert_allclose(curves[:2], archived, rtol=0., atol=3e-6)
                for tau in p['calibration'].values():
                    np.testing.assert_array_equal(curves[:2]>=tau, archived>=tau)
                maxima = np.maximum(maxima, (e, d, np.max(np.abs(curves[:2]-archived))))
                if rank in (21, 120, 360):
                    end = min(550, len(t))
                    keep = release<end
                    prefix = {k:signal[k][:end] for k in ('values', 'available')}
                    short, *_ = predict_shot(prefix, args, decoder, path, weights, parent[:end],
                                             release[keep], eligible[keep], dose[:end])
                    np.testing.assert_allclose(short, curves[:, :end], rtol=0., atol=3e-6)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(short>=tau, curves[:, :end]>=tau)
                    witnesses[str(sid)] = end
            dest = out/'shots'/name
            with dest.open('xb') as f:
                np.savez_compressed(f, fixed=fixed, bias_only=parent, release=release, eligible=eligible,
                                    dose=dose, **dict(zip(NEW, curves, strict=True)))
            hashes[name] = sha(dest)
            if rank==21 or rank%60==0:
                print(dict(stage='coefficient_carry_seal', rank=rank, seconds=round(time.monotonic()-started, 1)),
                      flush=True)
    if before!=[state_digest(m) for m in args[:2]]:
        raise ValueError('frozen model changed')
    write(out/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses, matched_endpoints=endpoints,
        max_parent_error=maxima[0], max_token_norm_error=maxima[1], max_reset_error=maxima[2],
        seconds=time.monotonic()-started))


def score(out):
    p, s = (read(out/k) for k in ('protocol.json', 'seal.json'))
    if (len(s['curve_sha256'])!=360 or tuple(p['arms'])!=ARMS
            or sha(out/'executed_probe.py')!=p['executed_sha256']
            or sha(out/'executed_group.py')!=p['group_sha256']
            or p['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or p['historical_seal']!=sha(HISTORY/'seal.json')):
        raise ValueError('incomplete or changed prediction seal')
    earlier = read(HISTORY/'per_shot.json')
    legacy = read(group.REFERENCE/'per_shot.json')
    per = {b:{k:[] for k in ARMS} for b in p['calibration']}
    for rank, sid in enumerate(p['order'], 1):
        name = f'{rank:04d}_{sid}.npz'
        f = out/'shots'/name
        if sha(f)!=s['curve_sha256'][name]:
            raise ValueError('sealed prediction changed')
        y, down, t = group.hdf._raw_truth(sid)
        with np.load(f, allow_pickle=False) as z:
            dormant = z['dose']==0.
            for k in NEW:
                np.testing.assert_array_equal(z[k][dormant], z[group.bias.BIAS][dormant])
            for band, tau in p['calibration'].items():
                previous = {k:legacy[band][k][rank-1] for k in group.OLD}
                previous.update({k:earlier[band][old][rank-1] for k,old in zip(NEW[:2], group.INNOVATION[-2:])})
                for k in ARMS:
                    row = dict(shot=sid, disruptive=y, down_s=down, **score_curve(
                        t, z[k], y, down, p['original'] if rank<=p['startup'] else tau))
                    if k in previous and row!=previous[k]:
                        raise ValueError('a frozen control changed its first alarm')
                    per[band][k].append(row)
    write(out/'per_shot.json', per)
    write(out/'lead_sensitivity.json', report([out/'per_shot.json'], ARMS))
    metrics = {b:{k:timing_decomposition(v) for k,v in rows.items()} for b,rows in per.items()}
    write(out/'result.json', dict(status='COMPLETED_COEFFICIENT_CARRY_DEVELOPMENT', metrics=metrics,
        paired={b:{f'{a}__to__{k}':flips(rows[a], rows[k]) for a,k in PAIRS} for b,rows in per.items()},
        raw_verified=360*len(ARMS)*len(p['calibration'])))
    print({b:{k:(v['tp'], v['fp']) for k,v in m.items()} for b,m in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.out.resolve())
    score(args.out.resolve())
