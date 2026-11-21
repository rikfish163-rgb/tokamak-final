"""Shot-local cross-diagnostic lag filters before the frozen patch encoder."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_persistence_20261111 as persistent
from . import commissioning_persistence_learning_20261111 as history
from . import commissioning_radial_east_20261111 as parent
from . import commissioning_split_learning_20261111 as direct
from . import io23, simple_data
from .commissioning_event_roughness_20261110 import COLUMNS
from .commissioning_transfer_20261108 import features
from .commissioning_transition_east_20261110 import scores
from .commissioning_weak_shape_data_20261111 import GROUPS
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


CHANNELS = tuple(COLUMNS)
GROUP = torch.tensor(GROUPS)
CROSS = GROUP[:, None] != GROUP
ORDERED = torch.nonzero(CROSS.triu(1), as_tuple=True)
W, LAG, RADIUS, INPUT_RMS = 128, 4, 1., .10
H, STRIDE, FIRST = direct.HORIZON, direct.STRIDE, direct.FIRST
FIRST_EFFECT = FIRST + H + 1
BASES, STARTUP = parent.BASES, parent.STARTUP
TRUE, SHIFT, HOLD = 'lag_true', 'lag_phase_shifted', 'lag_hold_previous'
ARMS = (*BASES, TRUE, SHIFT, HOLD)
PAIR = (('inherited', TRUE), (BASES[-1], TRUE), (SHIFT, TRUE), (HOLD, TRUE))


def patch_input(patches, theta):
    """Change trajectories in the already available W128, not their level or mask."""
    x = patches.reshape(-1, W, 46)
    cols = list(CHANNELS)
    z, m = x[..., cols], x[..., [23+c for c in cols]].bool()
    # Source motion over eight points ending four points before the recipient.
    valid = m[:, 8:-4] & m[:, :-12]
    d = (z[:, 8:-4] - z[:, :-12]).tanh().masked_fill(~valid, 0.)
    cross = CROSS.to(device=theta.device, dtype=theta.dtype)
    correction = d @ (theta * cross)
    supported = (valid.to(theta.dtype) @ cross) > 0.
    usable = m[:, 12:] & supported
    usable[:, -1] = False
    # Center the correction over the trailing window available at the
    # scored endpoint; this is not an estimate at an earlier token's time.
    mu = correction.masked_fill(~usable, 0.).sum(1) / usable.sum(1).clamp_min(1)
    correction = (correction - mu[:, None]).masked_fill(~usable, 0.)
    # Smooth, exact-zero normalization; fix an RMS ceiling in source units.
    # The eps only keeps the derivative at zero well defined.
    rms = torch.sqrt(correction.square().mean((1, 2), keepdim=True) + 1e-12)
    correction = correction / (rms / INPUT_RMS).clamp_min(1.)
    out = x.clone()
    out[:, 12:, cols] = z[:, 12:] + correction
    return out.reshape_as(patches)


def motion(y, *, shift=False):
    d = (y[:, LAG:] - y[:, :-LAG]).tanh()
    d = d / (d.square().mean(1, keepdim=True) + .02).sqrt()
    if shift:
        d = torch.stack([d[..., j].roll(5 * group, dims=1)
                         for j, group in enumerate(GROUPS)], dim=-1)
    return d


def ordered_signature(d):
    i, j = (v.to(d.device) for v in ORDERED)
    a, b = d[:, :-LAG, i], d[:, LAG:, j]
    c, e = d[:, :-LAG, j], d[:, LAG:, i]
    fwd, rev = (a*b).mean(1), (c*e).mean(1)
    return torch.cat(((fwd + rev) * .5, (fwd - rev) * .5), dim=1)


def matched_steps(g, shifted):
    n, s = (v.norm(dim=(1, 2), keepdim=True) for v in (g, shifted))
    common = (n > 1e-8) & (s > 1e-8)
    r = torch.where(common, torch.full_like(n, RADIUS), 0.)
    return -g * (r / n.clamp_min(1e-8)), -shifted * (r / s.clamp_min(1e-8))


def delayed_slots(q, ends):
    slots = np.searchsorted(q + H + 1, ends, side='right')
    return slots, np.maximum(slots-1, 0)


def _windows(base, signal, indices):
    windows = [simple_data.window(signal, int(t), base.cfg.window) for t in indices]
    return [torch.as_tensor(np.stack(a), device=base.center.device) for a in zip(*windows, strict=True)]


def _features(base, x, mask, theta):
    saved = []
    transform = base.embedding.register_forward_pre_hook(
        lambda _m, p: (patch_input(p[0], theta),))
    capture = base.risk_head.register_forward_pre_hook(lambda _m, p: saved.append(p[0]))
    try:
        base(x, mask)
    finally:
        capture.remove()
        transform.remove()
    return saved[0]


def _predict(hidden, prior, decoder):
    center, scale, w, b = decoder
    return prior + (((hidden - center) / scale) @ w.T + b).reshape(-1, H, len(CHANNELS))


def feedback(base, signal, tr, forecast, noise, decoder):
    """Estimate both directions from one shared zero-state diagnostic forecast."""
    q = tr['q']
    eligible = tr['observed'].all((1, 2)) & (q + H + 1 < tr['length'])
    q, dy = q[eligible], direct.normalized_target(tr, noise)[eligible]
    prior = direct.base_forecast(tr, forecast)[eligible]
    device = base.center.device
    n = len(CHANNELS)
    states = np.zeros((2, len(q)+1, n, n), np.float32)
    trace = np.zeros((len(q), 3), np.float32)
    for start in range(0, len(q), 128):
        end = start + 128
        x, m = _windows(base, signal, q[start:end])
        w = torch.zeros(len(x), n, n, device=device, requires_grad=True)
        hidden = _features(base, x, m, w)
        y, z = (torch.from_numpy(a[start:end]).to(device) for a in (dy, prior))
        p = ordered_signature(motion(_predict(hidden, z, decoder)))
        real, control = (ordered_signature(motion(y, shift=b)) for b in (False, True))
        lr, ls = (F.smooth_l1_loss(p, t, beta=.2, reduction='none').mean(1)
                  for t in (real, control))
        g, gs = (torch.autograd.grad(loss.sum(), w, retain_graph=i == 0)[0]
                 for i, loss in enumerate((lr, ls)))
        a, b = matched_steps(g, gs)
        states[:, start+1:end+1] = torch.stack((a, b)).cpu().numpy()
        trace[start:end] = torch.stack((g.norm(dim=(1, 2)), gs.norm(dim=(1, 2)), lr.detach()), 1).cpu().numpy()
    if not (np.isfinite(states).all() and np.isfinite(trace).all()):
        raise FloatingPointError('nonfinite lag adaptation')
    return q, states, trace, prior, dy


@torch.no_grad()
def _embed_active(base, signal, h, params):
    h = h.copy()
    use = np.flatnonzero(params.any((1, 2)))
    for start in range(0, len(use), 256):
        ids = use[start:start+256]
        x, m = _windows(base, signal, ids)
        theta = torch.from_numpy(params[ids]).to(x.device)
        h[ids] = _features(base, x, m, theta).cpu().numpy()
    return h


def encode(signal, args, decoder):
    base, net, params, old, quiet, mapper, coef, _ = args
    n = len(signal['values'])
    ends = np.arange(n)
    tr, inherited, ready = history.observed_track(net, signal, ends, params, old)
    state, _ = persistent.replay(*ready, quiet)
    latest = inherited + torch.einsum('...k,...f,kfw->...w', torch.from_numpy(tr['key']),
                     4.*torch.from_numpy(state['current_equal_norm']), mapper)
    x, _ = features(base, signal, 'cuda')
    h = x[:, :base.cfg.dim]
    views = {k: v.tanh().numpy() for k, v in zip(
        BASES, (torch.from_numpy(tr['pre']), inherited, latest), strict=True)}
    curves = scores(x, views, coef, BASES)
    q, matrices, log, prior, target = feedback(base, signal, tr, params[2], params[3], decoder)
    ids, prev = delayed_slots(q, ends)
    stat = np.zeros((len(q), 4), np.float32)
    for i, (name, table, slot) in enumerate(((TRUE, 0, ids), (SHIFT, 1, ids), (HOLD, 0, prev))):
        effective = matrices[table, slot]
        adapted = _embed_active(base, signal, h, effective)
        raw = np.concatenate((adapted, x[:, base.cfg.dim:]), axis=1)
        curves.update(scores(raw, {name: views[BASES[-1]]}, coef, (name,)))
        inactive = ~effective.any((1, 2))
        np.testing.assert_array_equal(curves[name][inactive], curves[BASES[-1]][inactive])
        with torch.no_grad():
            z, hp, y = (torch.from_numpy(a).cuda() for a in (prior, adapted[q], target))
            true_sig = ordered_signature(motion(y))
            p = ordered_signature(motion(_predict(hp, z, decoder)))
            stat[:, i+1] = F.smooth_l1_loss(p, true_sig, reduction='none', beta=.2).mean(1).cpu().numpy()
    stat[:, 0] = log[:, 2]
    return curves, dict(release=q+H+1, grad_norm=log[:, :2], next_phase_error=stat,
                        radius=np.linalg.norm(matrices[:, 1:], axis=(2, 3)).T)


def _bind_forecast(folder, p):
    fp = read(folder/'protocol.json')
    if (fp['source_sha256'] != p['source_sha256'] or fp['order'] != p['order']
            or fp['source_fit']['allowed'] != io23.source_roles()['train']
            or fp['calibration'] != p['calibration']
            or sha(folder/'forecast.npz') != fp['forecast_sha256']):
        raise ValueError('fixed J-TEXT-only auxiliary forecast required')
    with np.load(folder/'forecast.npz', allow_pickle=False) as f:
        return tuple(torch.from_numpy(f[k]).cuda() for k in ('center', 'scale', 'weight', 'bias'))


def source_check(args, decoder, output):
    """Ended-source one-step shadow, not J-TEXT online deployment or adaptation."""
    base, net, params, *_ = args
    center, scale, forecast, noise, _ = params
    ids = io23.source_roles()['val']
    advanced = io23.advanced_times()
    records, blocks, active = [], 0, 0
    for sid in ids:
        raw = simple_data.read_shot('J-TEXT', sid)
        tr = direct.track(net, raw, center, scale, np.empty(0, dtype=np.int64))
        q, theta, _, _, _ = feedback(base, raw, tr, forecast, noise, decoder)
        ends = q+H+1
        if not len(q):
            continue
        with torch.no_grad():
            x, mask = _windows(base, raw, ends)
            controls = (0.*theta[0, 1:], theta[0, 1:], theta[1, 1:])
            z = np.stack([base.risk_head(_features(base, x, mask, torch.from_numpy(w).cuda()))
                          .flatten().cpu().numpy() for w in controls], axis=1)
        truth = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
        d, down = truth['is_disrupt'], truth['down']
        timely = (np.round((down - raw['times'][ends])*1000., 9) > 10.) if d else np.ones(len(q), bool)
        keep = truth['valid'][ends] & timely
        if d:
            keep &= truth['y'][ends].astype(bool)
        blocks += int(keep.sum())
        action = theta[0, 1:].any((1, 2))
        active += int((keep & action).sum())
        if keep.any():
            diff = (z[keep, 1:] - z[keep, :1]) * (1. if d else -1.)
            records.append(dict(shot=sid, disruptive=d, endpoints=ends[keep].tolist(),
                                logit_gain_for_truth=diff.tolist()))
    if not active:
        raise ValueError('source VAL shadow found no observable common actions')
    summary = {}
    for label in (False, True):
        v = np.asarray([np.mean(r['logit_gain_for_truth'], axis=0)
                        for r in records if r['disruptive']==label])
        summary['positive' if label else 'safe'] = dict(shots=len(v), mean=v.mean(0).tolist(),
                                                          median=np.median(v, axis=0).tolist())
    write(output/'source_shadow.json', dict(scope='ENDED_JTEXT_VAL_LOCAL_MECHANISM_ONLY',
        columns=[TRUE, SHIFT], summary=summary, blocks=blocks, active=active, records=records))
    print(dict(stage='source_VAL160_shadow', blocks=blocks, active=active, summary=summary), flush=True)


def run(reference, coordinates, forecast, output):
    configure()
    started = time.monotonic()
    lock = load_inputs()
    ids = lock['ids']
    p, prior, args, assets = parent._frozen(reference, coordinates, lock)
    decoder = _bind_forecast(forecast, p)
    frozen = tuple(state_digest(m) for m in args[:2])
    cutoffs, original = p['calibration'], p['original']
    producer = Path(__file__)
    code = dict(p['code_sha256'], **{f.name: sha(f) for f in (producer, Path(parent.__file__))})
    (output/'shots').mkdir(parents=True)
    archive = output/'executed_code'
    archive.mkdir()
    for name, value in code.items():
        copyfile(producer.parent/name, archive/name)
        if sha(archive/name) != value:
            raise ValueError(f'archived producer changed: {name}')
    write(output/'protocol.json', dict(
        scope='EAST360_PREEMBEDDING_CAUSAL_LAG_MIXER_DEVELOPMENT', order=ids,
        startup=STARTUP, original=original, calibration=cutoffs, source_sha256=p['source_sha256'],
        source_check_ids=io23.source_roles()['val'], reference=str(reference),
        source_decoder=str(forecast/'forecast.npz'), decoder_sha256=sha(forecast/'forecast.npz'),
        assets=assets, code_sha256=code, arms=ARMS, comparisons=PAIR, first_effect=FIRST_EFFECT,
        window=H, stride=STRIDE, radius=RADIUS, input_rms_ceiling=INPUT_RMS,
        feedback='whole observed H25; shared zero-state forecast; gradient in source embedding input',
        phase_control='within each diagnostic group, cyclic shift of normalized motion at fixed lag',
        hold_control='previous mature matrix, skipping the newest response at the same endpoint',
        outcomes='EAST only after all 360 curves seal', persistent_updates=0))
    source_check(args, decoder, output)
    hashes, witnesses = {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST truth')):
        for rank, sid in enumerate(ids, 1):
            name = f'{rank:04d}_{sid}.npz'
            ref = reference/'shots'/name
            if sha(ref) != prior['curve_sha256'][name]:
                raise ValueError('fixed EAST reference curve changed')
            with np.load(ref, allow_pickle=False) as f:
                times, static = f['times'], {k: f[k] for k in BASES}
            trace = {}
            if rank <= STARTUP:
                curves = dict.fromkeys(ARMS, static['fixed'])
            else:
                if sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
                    raise ValueError('EAST signal file changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'], times)
                signal = prepare_actual(raw, rank, lock['mapping'])
                curves, trace = encode(signal, args, decoder)
                for k in BASES:
                    np.testing.assert_array_equal(curves[k], static[k])
                if rank in (21, 120, 360):
                    end = FIRST_EFFECT + STRIDE*3 + 5
                    cut, short = encode({k: signal[k][:end] for k in ('values', 'available')}, args, decoder)
                    for k in ARMS:
                        np.testing.assert_allclose(cut[k], curves[k][:end], atol=3e-6, rtol=0.)
                        for tau in cutoffs.values():
                            np.testing.assert_array_equal(cut[k]>=tau, curves[k][:end]>=tau)
                    np.testing.assert_array_equal(short['release'], trace['release'][trace['release'] < end])
                    witnesses[str(sid)] = end
            if any(not np.isfinite(v).all() for v in curves.values()):
                raise FloatingPointError('nonfinite EAST risk score')
            out = output/'shots'/name
            with out.open('xb') as f:
                np.savez_compressed(f, times=times, **curves, **trace)
            hashes[name] = sha(out)
            if rank % 40 == 0 or rank == STARTUP+1:
                print(dict(stage='EAST_sealed', rank=rank, seconds=round(time.monotonic()-started, 2)),
                      flush=True)
    if tuple(state_digest(m) for m in args[:2]) != frozen:
        raise ValueError('fixed networks changed')
    write(output/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses))
    per = {c: {k: [] for k in ARMS} for c in cutoffs}
    loss, count, updates = np.zeros(4), 0, 0
    for rank, sid in enumerate(ids, 1):
        name = f'{rank:04d}_{sid}.npz'
        out = output/'shots'/name
        if sha(out) != hashes[name]:
            raise ValueError('EAST risk curve not sealed')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(out, allow_pickle=False) as f:
            if rank > STARTUP:
                loss += f['next_phase_error'].sum(0)
                count += len(f['release'])
                updates += np.count_nonzero(f['radius'][:, 0])
            for c, group in per.items():
                tau = original if rank <= STARTUP else cutoffs[c]
                for k in ARMS:
                    group[k].append(dict(shot=sid, disruptive=d, down_s=down,
                                         **score_curve(f['times'], f[k], d, down, tau)))
    old = read(reference/'per_shot.json')
    if any(per[c][k] != old[c][k] for c in cutoffs for k in BASES):
        raise ValueError('fixed parent alarm scoring changed')
    write(output/'per_shot.json', per)
    metrics = {c: {k: timing_decomposition(v) for k, v in g.items()} for c, g in per.items()}
    paired = {c: {f'{a}__to__{b}': flips(g[a], g[b]) for a, b in PAIR} for c, g in per.items()}
    write(output/'lead_sensitivity.json', report([output/'per_shot.json'], ARMS))
    write(output/'result.json', dict(status='COMPLETED_CAUSAL_LAG_MIXER_DEVELOPMENT',
        metrics=metrics, paired=paired, mature_blocks=count, updates=int(updates),
        next_phase_mean=dict(zip(('zero', TRUE, SHIFT, HOLD), (loss/count).tolist(), strict=True)),
        seconds=time.monotonic()-started))
    print({c: {k: (m['tp'], m['fp']) for k, m in g.items()} for c, g in metrics.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'coordinates', 'forecast', 'output'):
        ap.add_argument('--'+name, required=True, type=Path)
    cli = ap.parse_args()
    run(cli.reference.resolve(), cli.coordinates.resolve(), cli.forecast.resolve(), cli.output.resolve())
