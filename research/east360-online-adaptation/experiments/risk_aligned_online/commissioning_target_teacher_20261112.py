"""Prequential EAST risk-direction teaching from sealed hard and safe discharges."""

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
from .commissioning_eos_replay_20261111 import seal_and_reveal
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


lag, layer = group.lag, group.layer
HISTORY = io23.ROOT/'probe_outputs/commissioning_signed_innovations_east360_20261112_v1'
INTRA = 'intra_innovation'
INTER, JOINT_RAW, JOINT, JOINT_OLD = NEW = ('inter_context', 'joint_raw', 'joint_taught', 'joint_taught_earlier')
ARMS = (*group.OLD, INTRA, *NEW)
PAIRS = ((group.bias.BIAS, INTRA), (group.bias.BIAS, INTER), (INTER, JOINT_RAW),
         (INTER, JOINT), (JOINT_RAW, JOINT), (JOINT_OLD, JOINT), (INTRA, JOINT))
RIDGE, ANCHORS = .05, 4


def unit_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def fit_bank(bank, rank):
    """Shot-balanced two-stage ridge; fit never receives the current outcome."""
    if any(r['rank']>=rank for r in bank):
        raise ValueError('only previous finalized shots may teach a prediction')
    counts = {k: sum(r['positive']==k for r in bank) for k in (True, False)}
    if not all(counts.values()):
        return None
    x, g, y = (np.concatenate([r[k] for r in bank]).astype(np.float64)
               for k in ('context', 'feedback', 'desired'))
    g = unit_rows(g)
    mass = np.concatenate([np.full(len(r['context']), .5/(counts[r['positive']]*len(r['context'])))
                           for r in bank])[:, None]

    def ridge(a, b):
        return np.linalg.solve(a.T@(mass*a)+RIDGE*np.eye(a.shape[1]), a.T@(mass*b))

    context = ridge(x, y)
    expected = ridge(x, g)
    feedback = ridge(g-x@expected, y-x@context-g)
    return context, expected, feedback


def forecast_directions(x, g, earlier, model):
    context, expected, feedback = model
    slow, typical = x@context, x@expected
    a, b = unit_rows(g), unit_rows(earlier)
    raw = slow+a
    return slow, raw, raw+(a-typical)@feedback, slow+b+(b-typical)@feedback


def teaching_points(times, score, dose, disruptive, down, *, timely):
    """Fixed parent peaks for safety, or timely near-window peaks on prior misses."""
    if disruptive and timely:
        return np.empty(0, dtype=np.int64)
    valid = dose>0.
    if disruptive:
        lead = np.round((down-times)*1000., 9)
        valid &= (lead>30.) & (lead<=125.)
    return np.asarray([p[np.argmax(score[p])] for p in np.array_split(np.flatnonzero(valid), ANCHORS)
                       if len(p)], dtype=np.int64)


def released_context(signal, base, z, q):
    """Risk and normalized current/field context, available at the release index."""
    release = q+lag.H+1
    ids = [io23.RAW23.index(k) for k in ('ip', 'Bt')]
    x = torch.as_tensor(signal['values'][:, ids], device=z.device)
    m = torch.as_tensor(signal['available'][:, ids], device=z.device)
    v = torch.asinh((x-base.center[ids])/base.scale[ids]).masked_fill(~m, 0.)
    prev = np.maximum(release-15, 0)
    rate = (v[release, 0]-v[prev, 0]).masked_fill(~(m[release, 0]&m[prev, 0]), 0.)
    features = torch.column_stack((torch.ones_like(rate), (z[release]/4.).tanh(),
                                  v[release, 0].tanh(), v[release, 1].tanh(), rate.tanh()))
    return features.cpu().numpy()


def forecast_track(signal, args, decoder, path, teacher, old):
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path)
    base = args[0]
    eligible, fields, waveform = group.feedback_waveform(signal, q, args[2], beta.device)
    states, common = group._estimate_innovation(base, signal, q, eligible, phase, correction, prior,
                                               fields, waveform, decoder, teacher)
    np.testing.assert_array_equal(q+lag.H+1, old['release'])
    np.testing.assert_array_equal(common, old['eligible'])
    return q, h, z, beta, phase, correction, states[-2:], common, released_context(signal, base, z, q)


@torch.no_grad()
def predict(signal, base, track, parent, old, learned):
    curves = np.stack((parent, old['innovation_current'], old['innovation_current'], old['innovation_lagged']))
    q, h, z, beta, phase, correction, g, common, x = track
    budget = old['dose']
    touched = np.flatnonzero(budget>0.)
    if learned is None or not len(touched):
        return curves, 0., 0.
    a, b = (v[1:].flatten(1).cpu().numpy() for v in g)
    directions = torch.as_tensor(np.stack(forecast_directions(x, a, b, learned)),
                                 device=z.device, dtype=z.dtype).unflatten(-1, g.shape[-2:])
    directions *= torch.as_tensor(common, device=z.device)[None, :, None, None]
    directions = torch.cat((torch.zeros_like(directions[:, :1]), directions), dim=1)
    parity = error = 0.
    for start in range(0, len(touched), group.BATCH):
        i = touched[start:start+group.BATCH]
        slot, _ = lag.delayed_slots(q, i)
        frac = group.handoff_fraction(q[slot-1]+lag.H+1, i, base.cfg.patch)
        x, m = lag._windows(base, signal, i)
        view, _, err = group.forward_hidden(base, x, m, layer.patcher(phase[slot], correction[slot]),
            writes=directions[:, slot],
            handoff=(directions[:, slot-1], torch.as_tensor(frac, device=z.device, dtype=z.dtype)),
            target_norm=torch.as_tensor(budget[i], device=z.device, dtype=z.dtype))
        p = (z[i]+((view-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
        parity = max(parity, float(np.max(np.abs(p[0]-parent[i]))))
        error = max(error, float(err))
        curves[:, i] = p[1:]
    if parity>3e-6 or error>1e-5 or not np.isfinite(curves).all():
        raise ValueError(f'archived norm or parent parity failed: {error:g}/{parity:g}')
    return curves, parity, error


def teaching_entry(signal, base, track, times, old, rank, sid, disruptive, down, threshold):
    q, h, z, beta, phase, correction, g, _, context = track
    history = old['innovation_current']
    timely = score_curve(times, history, disruptive, down, threshold)['timely']
    ends = teaching_points(times, history, old['dose'], disruptive, down, timely=timely)
    if not len(ends):
        return None
    slot, _ = lag.delayed_slots(q, ends)
    x, mask = lag._windows(base, signal, ends)
    u = torch.zeros_like(g[0, slot], requires_grad=True)
    view = group.forward_hidden(base, x, mask, layer.patcher(phase[slot], correction[slot]), state=u)
    desired, = torch.autograd.grad(((view-h[ends])*beta).sum(), u)
    direction = F.normalize(desired.flatten(1), dim=1, eps=1e-8)*(1. if disruptive else -1.)
    if not torch.isfinite(direction).all():
        raise FloatingPointError('nonfinite risk sensitivity on sealed shot')
    return dict(rank=rank, shot=sid, positive=disruptive, context=context[slot-1],
                feedback=g[0, slot].flatten(1).cpu().numpy(), desired=direction.cpu().numpy(),
                endpoints=ends.tolist(), releases=(q[slot-1]+lag.H+1).tolist(),
                parent_score=history[ends].tolist())


def run(out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    ref, sealed, args, decoder, path = group.load_reference(lock)
    old_protocol, old_seal = (read(HISTORY/k) for k in ('protocol.json', 'seal.json'))
    asset = HISTORY/'teaching.json'
    fixed = ('order', 'startup', 'original', 'calibration', 'source_sha256')
    if (any(ref[k] != old_protocol[k] for k in fixed)
            or old_protocol['scope'] != 'EAST360_SAME_SHOT_FORCE_INNOVATION_DEVELOPMENT'
            or tuple(old_protocol['arms']) != (*group.OLD, *group.INNOVATION)
            or old_protocol['reference_seal'] != sha(group.REFERENCE/'seal.json')
            or old_protocol['teaching_sha256'] != sha(asset)
            or len(old_seal['curve_sha256']) != 360):
        raise ValueError('target replay requires the original sealed within-shot reference')
    base = args[0]
    before = [state_digest(m) for m in args[:2]]
    teacher = torch.as_tensor(read(asset)['weights'], device=base.center.device, dtype=torch.float32)
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__), out/'executed_probe.py')
    copyfile(Path(group.__file__), out/'executed_group.py')
    p = dict(scope='EAST360_NATURAL_REVEAL_TARGET_DIRECTION_TEACHING', **{k:ref[k] for k in fixed},
        arms=ARMS, reference=str(group.REFERENCE), reference_seal=old_protocol['reference_seal'],
        historical_reference=str(HISTORY), historical_seal=sha(HISTORY/'seal.json'),
        executed_sha256=sha(out/'executed_probe.py'), group_sha256=sha(out/'executed_group.py'),
        teaching_sha256=sha(asset), ridge=RIDGE, supports_per_shot=ANCHORS,
        context=('intercept', 'released parent logit', 'released ip', 'released Bt', 'past ip change'),
        feedback='seal each whole shot before the natural HDF5 reveal; no active queries; no current outcome',
        memory='equal mass on previous hard-positive and safe shots, uniform per shot within class',
        writing='same realized token L2 per endpoint as the sealed intra arm, same 8-point handoff',
        strong_parent='inherited causal path frozen as reference, not the strictly static arm')
    write(out/'protocol.json', p)
    hashes, witnesses, timeline, bank, errors = {}, {}, [], [], []
    reader = io23.read_outcome
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome forbidden in prediction')):
        for rank, sid in enumerate(p['order'], 1):
            name = f'{rank:04d}_{sid}.npz'
            raw = io23.path_for('EAST', sid)
            a, b = (root/'shots'/name for root in (group.REFERENCE, HISTORY))
            if (sha(raw) != lock['hashes'][str(sid)] or sha(a) != sealed['curve_sha256'][name]
                    or sha(b) != old_seal['curve_sha256'][name]):
                raise ValueError('fixed input or reference changed')
            with np.load(a, allow_pickle=False) as f, np.load(b, allow_pickle=False) as j:
                t, frozen, parent = (f[k] for k in ('times', 'fixed', group.bias.BIAS))
                old = {k:j[k] for k in ('release', 'eligible', 'dose', 'innovation_current', 'innovation_lagged')}
                np.testing.assert_array_equal(old['release'], f['release'])
            learned = fit_bank(bank, rank)
            activity = dict(rank=rank, shot=sid, history_pos=sum(r['positive'] for r in bank),
                            history_safe=sum(not r['positive'] for r in bank), teacher_active=learned is not None,
                            latest_teacher_rank=max((r['rank'] for r in bank), default=None))
            if rank<=p['startup']:
                curves = np.broadcast_to(parent, (len(NEW), len(t)))
            else:
                record = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(record['times'], t)
                signal = lag.prepare_actual(record, rank, lock['mapping'])
                track = forecast_track(signal, args, decoder, path, teacher, old)
                curves, e, d = predict(signal, base, track, parent, old, learned)
                errors.append((e,d))
                if rank in (60, 180, 360):
                    end = min(390, len(t))
                    keep = old['release']<end
                    cropped = {k:v[keep] if k in ('release','eligible') else v[:end] for k,v in old.items()}
                    prefix = {k:signal[k][:end] for k in ('values','available')}
                    partial = forecast_track(prefix, args, decoder, path, teacher, cropped)
                    short, *_ = predict(prefix, base, partial, parent[:end], cropped, learned)
                    np.testing.assert_allclose(short, curves[:, :end], rtol=0., atol=3e-6)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(short>=tau, curves[:, :end]>=tau)
                    witnesses[str(sid)] = end
            arrays = dict(fixed=frozen, bias_only=parent, intra_innovation=old['innovation_current'],
                          release=old['release'], dose=old['dose'], **dict(zip(NEW,curves,strict=True)))
            hashes[name], (y, down) = seal_and_reveal(out/'shots'/name, arrays, sid, reader)
            if rank>p['startup']:
                entry = teaching_entry(signal, base, track, t, old, rank, sid, y, down, p['calibration']['0.2'])
                if entry is not None:
                    bank.append(entry)
                    activity.update({k:entry[k] for k in ('positive','endpoints','releases','parent_score')})
            timeline.append(activity)
            if rank in (20, 28) or rank%60==0:
                print(dict(stage='target_teacher_seal', rank=rank, hard=sum(r['positive'] for r in bank),
                           safe=sum(not r['positive'] for r in bank), seconds=round(time.monotonic()-started,1)),
                      flush=True)
    if before != [state_digest(m) for m in args[:2]]:
        raise ValueError('base or inherited adaptation model changed')
    write(out/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses,
        max_parent_error=max(e for e,_ in errors), max_token_norm_error=max(d for _,d in errors),
        seconds=time.monotonic()-started))
    write(out/'timeline.json', timeline)
    score(out)


def score(out):
    p, s, timeline = (read(out/k) for k in ('protocol.json','seal.json','timeline.json'))
    if (len(s['curve_sha256']) != 360 or sha(out/'executed_probe.py') != p['executed_sha256']
            or sha(out/'executed_group.py') != p['group_sha256']
            or p['reference_seal'] != sha(group.REFERENCE/'seal.json')
            or p['historical_seal'] != sha(HISTORY/'seal.json')):
        raise ValueError('incomplete target replay seal')
    previous = read(HISTORY/'per_shot.json')
    legacy = read(group.REFERENCE/'per_shot.json')
    per = {band:{k:[] for k in ARMS} for band in p['calibration']}
    for rank, (sid, witness) in enumerate(zip(p['order'], timeline, strict=True), 1):
        if ((witness['shot'], witness['rank']) != (sid, rank)
                or (witness['latest_teacher_rank'] is not None and witness['latest_teacher_rank']>=rank)):
            raise ValueError('future or same-shot teacher entry')
        name = f'{rank:04d}_{sid}.npz'
        f = out/'shots'/name
        if sha(f)!=s['curve_sha256'][name]:
            raise ValueError('sealed prediction changed')
        y, down, t = group.hdf._raw_truth(sid)
        with np.load(f,allow_pickle=False) as z:
            dormant = z['dose']==0.
            for k in NEW:
                np.testing.assert_array_equal(z[k][dormant], z[group.bias.BIAS][dormant])
            for band, tau in p['calibration'].items():
                for k in ARMS:
                    row = dict(shot=sid, disruptive=y, down_s=down, **score_curve(
                        t, z[k], y, down, p['original'] if rank<=p['startup'] else tau))
                    if k==INTRA and row != previous[band]['innovation_current'][rank-1]:
                        raise ValueError('within-shot reference scoring changed')
                    if k in group.OLD and row != legacy[band][k][rank-1]:
                        raise ValueError('strong parent or static reference changed')
                    per[band][k].append(row)
    write(out/'per_shot.json', per)
    write(out/'lead_sensitivity.json', report([out/'per_shot.json'], ARMS))
    metrics = {band:{k:timing_decomposition(v) for k,v in rows.items()} for band,rows in per.items()}
    write(out/'result.json', dict(status='COMPLETED_TARGET_DIRECTION_TEACHING_DEVELOPMENT', metrics=metrics,
        paired={band:{f'{a}__to__{b}':flips(rows[a],rows[b]) for a,b in PAIRS} for band,rows in per.items()},
        raw_verified=360*len(ARMS)*len(p['calibration']), active_shots=sum(v['teacher_active'] for v in timeline)))
    print({b:{k:(r['tp'],r['fp']) for k,r in m.items()} for b,m in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out.resolve())
