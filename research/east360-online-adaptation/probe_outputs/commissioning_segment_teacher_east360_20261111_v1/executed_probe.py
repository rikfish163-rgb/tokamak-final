"""EAST commissioning: compare point and segment teaching of an internal fast adapter."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_eos_replay_20261111 as replay
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_lag_teacher_20261111 as teacher
from . import commissioning_receiver_reach_20261111 as reach
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


FIXED, REAL, SHIFT, POINT, POINT_SHIFT, SEGMENT, SEGMENT_SHIFT = ARMS = (
    'fixed', 'uniform_true', 'uniform_shifted', 'point_true', 'point_shifted',
    'segment_true', 'segment_shifted',
)
PAIRS = ((FIXED, REAL), (SHIFT, REAL), (FIXED, POINT), (POINT_SHIFT, POINT),
         (FIXED, SEGMENT), (REAL, SEGMENT), (POINT, SEGMENT), (SEGMENT_SHIFT, SEGMENT))
K, GROUPS, RADIUS = 5, 2, 2.


def select_blocks(times, z, slots, disruptive, down):
    """Best blocks from distinct released updates, scored on the frozen parent."""
    if len(times) < K:
        return np.empty((0, K), dtype=int)
    points = np.arange(len(times))
    valid = slots > 0
    if disruptive:
        lead = np.round((down-times)*1000., 9)
        valid &= (lead > 30.) & (lead <= 125.)
    b = np.lib.stride_tricks.sliding_window_view(points, K)
    b = b[valid[b].all(1) & (slots[b] == slots[b[:, :1]]).all(1)]
    rating = z[b].min(1) if disruptive else z[b].max(1)
    order = np.argsort(-rating, kind='stable')
    _, first = np.unique(slots[b[order, 0]], return_index=True)
    return b[order[np.sort(first)[:GROUPS]]]


def summarize_blocks(z, disruptive, sustained):
    point = z.amax((1, 2))
    if not sustained:
        return point
    return torch.where(disruptive, z.amin(2).amax(1), point)


def segment_loss(z, disruptive, boundary, sustained):
    return replay.alarm_loss(summarize_blocks(z, disruptive, sustained), disruptive, boundary)


def _patch_mask(base, x, mask):
    return (mask & torch.isfinite(x)).reshape(len(mask), -1, base.cfg.patch*23).any(-1)


def _writable(valid):
    pair = valid[:, 1:] & valid[:, :-1]
    return F.pad(pair, (1, 1))  # first patch and last (CLS) cannot be written


def token_change(tokens, valid_patch, vector):
    adjacent = (tokens[:, 1:-1] - tokens[:, :-2]).tanh()
    return (F.pad(adjacent, (0, 0, 1, 1))*vector[:, None]).masked_fill(
        ~_writable(valid_patch)[..., None], 0.)


def equal_dose(delta, cap):
    """Hold actual RMS fixed across arms for each endpoint, not just parameter norm."""
    norm = (delta.square().mean((2, 3)) + 1e-24).sqrt()
    common = norm.detach().amin(0).clamp_max(cap)
    common = torch.where(common > 1e-10, common, 0.)
    return delta * (common / norm)[..., None, None], common


def _forward(base, x, mask, transform):
    hidden = []
    a = base.encoder.layers[0].register_forward_hook(lambda _m, _p, out: out + transform(out))
    b = base.risk_head.register_forward_pre_hook(lambda _m, p: hidden.append(p[0]))
    try:
        base(x, mask)
    finally:
        b.remove()
        a.remove()
    return hidden[0]


def adapted_hidden(base, x, mask, directions, cap):
    arms, batch, dim = directions.shape
    valid = _patch_mask(base, x, mask).repeat(arms, 1)
    dose = []

    def transform(tokens):
        delta = token_change(tokens, valid, directions.flatten(0, 1)).reshape(
            arms, batch, tokens.shape[1], dim)
        offset, rms = equal_dose(delta, cap)
        dose.append(rms)
        return offset.flatten(0, 1)

    h = _forward(base, x.repeat(arms, 1, 1), mask.repeat(arms, 1, 1), transform)
    return h.reshape(arms, batch, dim), dose[0]


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _feedback(base, signal, q, prior, truth, decoder):
    """The two controls see identical complete observed spans and release times."""
    data = base.center.new_zeros(2, len(q)+1, base.cfg.dim)
    norms = base.center.new_zeros(len(q), 2)
    for start in range(0, len(q), 128):
        end = min(start+128, len(q))
        sl = slice(start, end)
        x, m = lag._windows(base, signal, q[sl])
        p, y = (torch.from_numpy(a[sl]).to(x.device) for a in (prior, truth))
        valid = _patch_mask(base, x, m)
        vector = x.new_zeros(len(x), base.cfg.dim).requires_grad_(True)
        hidden = _forward(base, x, m, lambda h: token_change(h, valid, vector))
        pred = lag.ordered_signature(lag.motion(lag._predict(hidden, p, decoder)))
        target = [lag.ordered_signature(lag.motion(y, shift=s)) for s in (False, True)]
        g = torch.stack([torch.autograd.grad(F.smooth_l1_loss(pred, a, beta=.2, reduction='sum'),
                         vector, retain_graph=i == 0)[0] for i, a in enumerate(target)])
        data[:, start+1:end+1] = -_unit(g)
        norms[sl] = g.norm(dim=2).T
    if not torch.isfinite(norms).all():
        raise FloatingPointError('nonfinite internal forecast gradient')
    return data, norms


def _cold_doses(base, signal, q, keys, p, y, decoder):
    if not len(q):
        return []
    i = np.unique(np.linspace(0, len(q)-1, 3, dtype=int))
    zero = base.center.new_zeros(4, replay.N, replay.N)
    step, _ = replay._feedback(base, signal, q[i], keys[i], p[i], y[i], decoder, zero)
    with torch.no_grad():
        x, m = lag._windows(base, signal, q[i]+lag.H+1)
        before, _ = reach._hidden(base, x, m, torch.zeros_like(step[0, 1:]))
        after, _ = reach._hidden(base, x, m, step[0, 1:])
        change = (after-before).masked_fill(~_writable(_patch_mask(base, x, m))[..., None], 0.)
        rms = change.square().mean((1, 2)).sqrt()
        return rms[rms > 0].tolist()


def _context(tr, args, signal, decoder):
    q, keys, p, y = teacher._mature(tr, args[2][2], args[2][3])
    direction, norm = _feedback(args[0], signal, q, p, y, decoder)
    key = F.pad(torch.from_numpy(keys).to(direction.device), (0, 0, 1, 0))
    return (q, key, direction, norm), (q, keys, p, y)


def _taught(g, keys, state):
    return _unit(g * (1. + 2.*(keys @ state).tanh()))


@torch.no_grad()
def _curves(base, signal, track, context, states, cap):
    _, h, z, beta = track
    q, key, direction, norms = context
    slots = replay.release_slots(q, np.arange(len(z)))
    p = z.sigmoid()
    curves = p.repeat(len(ARMS), 1)
    rms = torch.zeros_like(z)
    active = np.flatnonzero(slots)
    for start in range(0, len(active), 128):
        i = active[start:start+128]
        release = slots[i]
        x, m = lag._windows(base, signal, i)
        g = direction[:, release]
        dirs = torch.cat((g, *(_taught(g, key[release], a) for a in states)))
        out, dose = adapted_hidden(base, x, m, dirs, cap)
        curves[1:, i] = torch.where(dose > 0., (z[i] + (out-h[i]) @ beta).sigmoid(), p[i])
        rms[i] = dose
    return dict(zip(ARMS, curves.cpu().numpy(), strict=True)), dict(
        release=q+lag.H+1, grad_norm=norms.cpu().numpy(), token_rms=rms.cpu().numpy())


def _row(base, signal, times, track, context, rank, sid, d, down):
    tr, h, z, _ = track
    q, key, g, _ = context
    slots = replay.release_slots(q, np.arange(len(z)))
    values = z.cpu().numpy()
    b = select_blocks(times, values, slots, d, down)
    if not len(b):
        return None
    anchor = b[0, values[b[0]].argmax()]
    take = np.pad(b, ((0, GROUPS-len(b)), (0, 0)), mode='edge').ravel()
    x, m = lag._windows(base, signal, take)
    arrays = dict(x=x, m=m, h=h[take], z=z[take], key=key[slots[take]], g=g[0, slots[take]])
    return dict(rank=rank, shot=sid, disruptive=d, blocks=b.tolist(),
                match=np.r_[values[anchor], 2.*tr['key'][anchor, 1:]],
                **{k: v.cpu().numpy() for k, v in arrays.items()})


def _teach(base, beta, states, rows, boundary, cap):
    def batch(k):
        return torch.from_numpy(np.concatenate([r[k] for r in rows])).to(beta.device)

    x, m, h, z, key, g = (batch(k) for k in ('x', 'm', 'h', 'z', 'key', 'g'))
    pos = torch.tensor([r['disruptive'] for r in rows], device=beta.device)
    weights = [a.detach().requires_grad_(True) for a in states]
    direction = torch.stack([_taught(g, key, a) for a in weights])
    adapted, _ = adapted_hidden(base, x, m, direction, cap)
    logits = (z + (adapted-h) @ beta).reshape(2, len(rows), GROUPS, K)
    losses = torch.stack([segment_loss(v, pos, boundary, bool(i)) for i, v in enumerate(logits)])
    gradients = torch.autograd.grad(losses.sum(), weights)
    if not all(torch.isfinite(a).all() for a in (*gradients, losses)):
        raise FloatingPointError('nonfinite teacher update')
    with torch.no_grad():
        proposed = [a-replay.STEP*g/g.norm().clamp_min(1e-6)
                    for a, g in zip(weights, gradients, strict=True)]
        new = tuple(a * (RADIUS / a.norm().clamp_min(RADIUS)) for a in proposed)
        delta = [float((a-b).norm()) for a, b in zip(new, states, strict=True)]
    return new, dict(loss=losses.detach().tolist(), step=delta,
                     gradient=[float(a.norm()) for a in gradients])


def run(reference, output):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    ids = lock['ids']
    p, seal, previous, args, decoder = replay._load(reference, lock)
    base, net = args[:2]
    frozen = state_digest(base), state_digest(net)
    thresholds, original = p['calibration'], p['original']
    boundary = float(logit(thresholds['0.2']))
    states = tuple(base.center.new_zeros(4, base.cfg.dim) for _ in range(2))
    (output/'shots').mkdir(parents=True)
    copyfile(Path(__file__), output/'executed_probe.py')
    protocol = dict(scope='EAST360_NATURAL_EOS_INTERNAL_SEGMENT_TEACHING_DEVELOPMENT',
        order=ids, startup=lag.STARTUP, calibration=thresholds, original=original, arms=ARMS,
        reference=str(reference), reference_seal=sha(reference/'seal.json'),
        producer_sha256=sha(Path(__file__)), source_sha256=p['source_sha256'],
        feedback='whole shot sealed before HDF5 EOS; H25 observed before temporary update',
        effect='bounded patch motion write after first encoder layer, risk readout frozen',
        replay='same current shot and one prior nearest ended shot per class for both teachers',
        support='two five-consecutive-endpoint blocks in distinct released update slots; '
                'disruptive (30,125] ms, safe unrestricted; select on frozen parent',
        comparison='real and shifted forecasts use the same direction norm and per-endpoint actual '
                   'token RMS; phase shift uses identical available spans; safe teacher loss identical',
        step=replay.STEP, parameter_radius=RADIUS, temperature=replay.TEMPERATURE,
        stride=lag.STRIDE, first_effect=lag.FIRST_EFFECT)
    write(output/'protocol.json', protocol)
    reader = io23.read_outcome
    history, prepared, fit, hashes, witnesses = [], [], [], {}, {}
    per = {band: {k: [] for k in ARMS} for band in thresholds}
    cap = None
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome forbidden in prediction')):
        for rank, sid in enumerate(ids, 1):
            name = f'{rank:04d}_{sid}.npz'
            f = previous/'shots'/name
            if sha(f) != seal['curve_sha256'][name] or sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
                raise ValueError('sealed reference or current HDF5 changed')
            raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            times = raw['times']
            with np.load(f, allow_pickle=False) as saved:
                np.testing.assert_array_equal(times, saved['times'])
                old = saved[FIXED]
            if rank <= lag.STARTUP:
                curves, trace = dict.fromkeys(ARMS, old), {}
            else:
                signal = lag.prepare_actual(raw, rank, lock['mapping'])
                track = replay._track(signal, args)
                context, _ = _context(track[0], args, signal, decoder)
                curves, trace = _curves(base, signal, track, context, states, cap)
                np.testing.assert_array_equal(curves[FIXED], old)
                for k in ARMS[1:]:
                    np.testing.assert_array_equal(curves[k][trace['token_rms'] == 0],
                                                  old[trace['token_rms'] == 0])
                if rank in (21, 120, 360):
                    end = min(len(times), lag.FIRST_EFFECT+3*lag.STRIDE+5)
                    prefix = {k: signal[k][:end] for k in ('values', 'available')}
                    short_track = replay._track(prefix, args)
                    short_ctx, _ = _context(short_track[0], args, prefix, decoder)
                    short, st = _curves(base, prefix, short_track, short_ctx, states, cap)
                    for k in ARMS:
                        np.testing.assert_allclose(short[k], curves[k][:end], atol=3e-6, rtol=0.)
                        for tau in thresholds.values():
                            np.testing.assert_array_equal(short[k] >= tau, curves[k][:end] >= tau)
                    np.testing.assert_array_equal(st['release'], trace['release'][trace['release'] < end])
                    witnesses[str(sid)] = end
            if not all(np.isfinite(a).all() for a in curves.values()):
                raise FloatingPointError('nonfinite risk trace')
            hashes[name], (d, down) = replay.seal_and_reveal(output/'shots'/name,
                dict(times=times, **curves, **trace), sid, reader)
            for band, tau in thresholds.items():
                for k, v in curves.items():
                    per[band][k].append(dict(shot=sid, disruptive=d, down_s=down,
                         **score_curve(times, v, d, down, original if rank <= lag.STARTUP else tau)))
            if rank <= lag.STARTUP:
                prepared.append((rank, sid, raw, d, down))
                if rank == lag.STARTUP:
                    doses = []
                    for r, s, data, dy, dt in prepared:
                        signal = lag.prepare_actual(data, lag.STARTUP+1, lock['mapping'])
                        track = replay._track(signal, args)
                        context, inputs = _context(track[0], args, signal, decoder)
                        doses.extend(_cold_doses(base, signal, *inputs, decoder))
                        row = _row(base, signal, data['times'], track, context, r, s, dy, dt)
                        if row is not None:
                            history.append(row)
                    if not doses or {r['disruptive'] for r in history} != {True, False}:
                        raise ValueError('cold start lacks supported feedback or both ended outcomes')
                    cap = float(np.median(doses))
                    protocol.update(cap=cap, cap_samples=len(doses),
                        cap_basis='median first-layer patch RMS induced by prior input adapter at '
                                  'up to three releases per ended cold-start shot (no outcomes used)')
                    write(output/'protocol.json', protocol)
                    prepared.clear()
            else:
                row = _row(base, signal, times, track, context, rank, sid, d, down)
                if row is None:
                    fit.append(dict(rank=rank, shot=sid, seal=hashes[name], supported=False))
                else:
                    past = replay.select_history(history, row)[1]
                    states, stats = _teach(base, track[-1], states, [row, *past], boundary, cap)
                    fit.append(dict(rank=rank, shot=sid, seal=hashes[name], supported=True,
                        blocks=row['blocks'], replay_ranks=[r['rank'] for r in past], **stats))
                    history.append(row)
            if rank in (20, 21) or rank % 40 == 0:
                print(dict(stage='shot_sealed', rank=rank, cap=cap,
                           seconds=round(time.monotonic()-started, 2)), flush=True)
    if (state_digest(base), state_digest(net)) != frozen:
        raise ValueError('frozen source model changed')
    write(output/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses))
    write(output/'teaching.json', fit)
    write(output/'per_shot.json', per)
    write(output/'lead_sensitivity.json', report([output/'per_shot.json'], ARMS))
    result = dict(status='COMPLETED_EAST360_INTERNAL_SEGMENT_TEACHING_DEVELOPMENT',
        seconds=time.monotonic()-started,
        metrics={b: {k: timing_decomposition(v) for k, v in a.items()} for b, a in per.items()},
        paired={b: {f'{p}__to__{q}': flips(a[p], a[q]) for p, q in PAIRS} for b, a in per.items()})
    write(output/'result.json', result)
    print({b: {k: (m['tp'], m['fp']) for k, m in a.items()}
           for b, a in result['metrics'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path, default=replay.REFERENCE)
    cli = parser.parse_args()
    run(cli.reference, cli.output)
