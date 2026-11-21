"""Shot-local, diagnostic-group temporal exchanges before the first attention block."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_layer_transport_20261112 as layer
from . import commissioning_recovery_clock_20261111 as recovery
from . import io23, simple_data
from .commissioning_transition_data_20261110 import _window_mean
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


bias, lag, guard, teacher = layer.bias, layer.lag, layer.guard, layer.teacher
REFERENCE = layer.REFERENCE
EXCLUDED = tuple(io23.RAW23.index(k) for k in ('Mir_avg_amp', 'Mir_avg_fre'))
TRUSTED = tuple(tuple(c for c, g in zip(lag.CHANNELS, lag.GROUP.tolist(), strict=True)
                      if g == i and c not in EXCLUDED) for i in range(4))
POSITIONS = tuple(tuple(lag.CHANNELS.index(c) for c in g) for g in TRUSTED)
USE = sorted(c for g in POSITIONS for c in g)
TRUE, SHAM, HEURISTIC = NEW = ('group_gradient', 'group_swapped_gradient', 'group_residual_write')
TRANSPORT = ('origin_gradient', 'release_gradient', 'release_swapped_gradient')
RAMP = ('origin_ramped', 'release_ramped', 'release_swapped_ramped')
OLD = ('fixed', bias.BIAS)
ARMS = (*OLD, *NEW)
PAIRS = tuple((bias.BIAS, k) for k in NEW) + ((SHAM, TRUE), (HEURISTIC, TRUE))
BATCH, BIN, BETA, EPS = 128, 5, .2, 1e-8


def group_edges(patches, weight):
    """Use only pairwise observed value lanes, never missingness or spectral lanes."""
    x = patches.unflatten(-1, (-1, 46))
    w = weight.unflatten(-1, (-1, 46))
    edges = []
    for cols in TRUSTED:
        both = x[:, :-1, :, [c+23 for c in cols]].bool() & x[:, 1:, :, [c+23 for c in cols]].bool()
        change = (x[:, 1:, :, cols] - x[:, :-1, :, cols]).masked_fill(~both, 0.)
        edges.append(F.normalize(torch.einsum('besk,dsk->bed', change, w[..., cols]), dim=-1, eps=EPS))
    return torch.stack(edges, dim=2)


def exchange(coeff, edges):
    step = torch.einsum('...beg,begd->...bed', coeff, edges)
    return F.pad(step, (0, 0, 1, 0)) - F.pad(step, (0, 0, 0, 1))


def sham_error(error):
    return error.roll(1, dims=-1)


def feedback_write(error, edges):
    bins = torch.arange(edges, device=error.device)*error.shape[1]//edges
    return error[:, bins]


def radii(level, center, spread):
    avg = level.unflatten(1, (-1, BIN)).mean(2)
    relative = (avg - center[:, None])/spread[:, None]
    return torch.stack([torch.log1p((relative[..., g].square().mean(-1)+1e-12).sqrt())
                        for g in POSITIONS], dim=-1)


def matched_writes(delta, radius):
    norm = delta.flatten(2).norm(dim=-1)
    common = (norm > EPS).all(0)
    return delta*(radius*common/norm.clamp_min(EPS))[..., None, None]


def handoff_fraction(release, endpoints, patch):
    return np.clip((endpoints-release)/patch, 0., 1.)


def blend_writes(new, old, radius, fraction):
    a, b = matched_writes(new, radius), matched_writes(old, radius)
    x = fraction[None, :, None, None]
    blend = x*a+(1.-x)*b
    # Preserve the temporal blend's lower dose even if directions cancel.
    # Match to the smallest across controls, never to the full-dose ceiling.
    common = torch.minimum(blend.flatten(2).norm(dim=-1).amin(dim=0), radius)
    joined = matched_writes(blend, common)
    return torch.where(x==1., a, torch.where(x==0., b, joined))


def forward_hidden(base, x, mask, transform, *, state=None, writes=None, handoff=None):
    """Run the production encoder; state differentiates, writes compare equal doses."""
    n = len(x)
    count = 1 if writes is None else 1+len(writes)
    edges, radius, hidden, realized = [], [], [], []

    def prepare(_module, inputs):
        raw = inputs[0][:n]
        corrected = transform(raw)
        edges.append(group_edges(corrected, base.embedding.weight))
        if writes is not None:
            change = F.linear(corrected-raw, base.embedding.weight)
            radius.append(change.flatten(1).norm(dim=1))
        return (corrected.repeat(count, 1, 1),)

    def inject(_module, inputs):
        before = inputs[0]
        if writes is None:
            return (layer.add_to_patches(before, exchange(state, edges[0])), *inputs[1:])
        k, d = edges[0].shape[1]+1, before.shape[-1]
        raw = exchange(writes, edges[0])
        if handoff is None:
            delta = matched_writes(raw, radius[0])
        else:
            old, fraction = handoff
            delta = blend_writes(raw, exchange(old, edges[0]), radius[0], fraction)
        actual = before[n:, :k].reshape(count-1, n, k, d)
        got = ((actual+delta)-actual).flatten(2).norm(dim=-1)
        wanted = (torch.where(delta.flatten(2).norm(dim=-1)>0., radius[0], 0.) if handoff is None
                  else delta[0].flatten(1).norm(dim=1))
        err = torch.maximum((got-wanted).abs().max(), (got-radius[0]).clamp_min(0.).max())
        realized.append((got[0], err))
        delta = torch.cat((torch.zeros_like(delta[:1]), delta)).flatten(0, 1)
        return (layer.add_to_patches(before, delta), *inputs[1:])

    handles = (base.embedding.register_forward_pre_hook(prepare),
               base.encoder.layers[0].register_forward_pre_hook(inject),
               base.risk_head.register_forward_pre_hook(lambda _module, inputs: hidden.append(inputs[0])))
    try:
        base(x.repeat(count, 1, 1), mask.repeat(count, 1, 1))
    finally:
        for handle in handles:
            handle.remove()
    return hidden[0] if writes is None else (hidden[0].reshape(count, n, -1), *realized[0])


def _feedback_data(signal, q, parameters, device):
    center, scale, _, noise, _ = parameters
    v, m, _, _ = lag.direct.weak.shot_coordinates(signal, center, scale)
    ref = recovery.rolling_reference(v, m)
    recent, count = _window_mean(v, m, q, -15, 16)
    eligible = ref['reference_known'][q][:, USE].all(1) & (count[:, USE] >= 8).all(1)
    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)
    center, spread = tensor(ref['center'][q]), tensor(ref['spread'][q])
    truth = radii(tensor(v[q[:, None]+np.arange(1, lag.H+1)]), center, spread)
    return eligible, (tensor(recent), center, spread, tensor(noise)), truth


def _radius_from_hidden(h, prior, recent, center, spread, noise, decoder):
    return radii(lag._predict(h, prior, decoder)*noise + recent[:, None], center, spread)


def release_backprop(pred, issued_hidden, issued_state, released_hidden, released_state, error):
    """Keep the arrived error in issued-hidden coordinates; change its write Jacobian."""
    force = (error/BETA).clamp(-1., 1.)
    origin, real = torch.autograd.grad(pred, (issued_state, issued_hidden),
                                      grad_outputs=force, retain_graph=True)
    wrong, = torch.autograd.grad(pred, issued_hidden, grad_outputs=sham_error(force))
    a = torch.autograd.grad(released_hidden, released_state, grad_outputs=real, retain_graph=True)[0]
    b = torch.autograd.grad(released_hidden, released_state, grad_outputs=wrong)[0]
    return origin, a, b


def _estimate(base, signal, q, eligible, phase, correction, prior, fields, target, decoder,
              *, transport=False):
    edges = base.cfg.window//base.cfg.patch-1
    states = prior.new_zeros((len(NEW), len(q)+1, edges, len(TRUSTED)))
    slots, _ = lag.delayed_slots(q, q)
    eligible_ids = np.flatnonzero(eligible)
    recent, center, spread, noise = fields
    for start in range(0, len(eligible_ids), BATCH):
        i = eligible_ids[start:start+BATCH]
        x, mask = lag._windows(base, signal, q[i])
        state = states.new_zeros((len(i), edges, len(TRUSTED)), requires_grad=True)
        transform = layer.patcher(phase[slots[i]], correction[slots[i]])
        h = forward_hidden(base, x, mask, transform, state=state)
        pred = _radius_from_hidden(h, prior[i], recent[i], center[i], spread[i], noise, decoder)
        error = target[i] - pred.detach()
        if transport:
            xr, mr = lag._windows(base, signal, q[i]+lag.H+1)
            ready = torch.zeros_like(state, requires_grad=True)
            hr = forward_hidden(base, xr, mr, layer.patcher(phase[i+1], correction[i+1]), state=ready)
            states[:, i+1] = torch.stack(release_backprop(pred, h, state, hr, ready, error))
            continue
        diff = pred - pred.detach()
        def loss(delta):
            return F.smooth_l1_loss(diff, delta, beta=BETA, reduction='sum')
        real = -torch.autograd.grad(loss(error), state, retain_graph=True)[0]
        wrong = -torch.autograd.grad(loss(sham_error(error)), state)[0]
        states[:, i+1] = torch.stack((real, wrong, feedback_write(error, edges)))
    if not torch.isfinite(states).all():
        raise FloatingPointError('nonfinite first-layer feedback')
    return states


@torch.no_grad()
def _replay(base, signal, h, z, beta, q, eligible, phase, correction, states, archived,
            prior, fields, target, decoder, *, ramp=False):
    slots, _ = lag.delayed_slots(q, np.arange(len(z)))
    active = np.flatnonzero(slots)
    curves = np.broadcast_to(archived, (len(NEW), len(z))).copy()
    dose = np.zeros(len(z), dtype=np.float32)
    auxiliary = []
    parent_error, dose_error = 0., 0.
    recent, center, spread, noise = fields
    for start in range(0, len(active), BATCH):
        i = active[start:start+BATCH]
        s = slots[i]
        x, m = lag._windows(base, signal, i)
        handoff = None
        if ramp:
            fraction = handoff_fraction(q[s-1]+lag.H+1, i, base.cfg.patch)
            old = states[:, s-1]*torch.as_tensor(eligible[s-1], device=x.device)[None, :, None, None]
            handoff = old, torch.as_tensor(fraction, device=x.device, dtype=x.dtype)
        view, rms, err = forward_hidden(base, x, m, layer.patcher(phase[s], correction[s]),
                                        writes=states[:, s], handoff=handoff)
        p = (z[i] + ((view-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
        parent_error = max(parent_error, float(np.max(np.abs(p[0]-archived[i]))))
        dose_error = max(dose_error, float(err))
        dose[i] = rms.cpu().numpy()
        applied = dose[i] > 0.
        curves[:, i[applied]] = p[1:, applied]
        # Forecast at a later origin with the state already present at that origin.
        j = np.flatnonzero(np.isin(i, q[eligible]) & applied)
        if len(j):
            k = np.searchsorted(q, i[j])
            pred = [_radius_from_hidden(v[j], prior[k], recent[k], center[k], spread[k], noise, decoder)
                    for v in view]
            auxiliary.append(torch.stack([F.smooth_l1_loss(a, target[k], beta=BETA,
                                          reduction='none').mean((1, 2)) for a in pred], 1).cpu().numpy())
    if parent_error > 3e-6 or dose_error > 1e-5 or not np.isfinite(curves).all():
        raise ValueError(f'parent reproduction or first-layer equal dose failed: {parent_error:g}/{dose_error:g}')
    aux = np.concatenate(auxiliary) if auxiliary else np.empty((0, len(NEW)+1), dtype=np.float32)
    return curves, dose, aux, parent_error, dose_error


def parent_track(signal, args, decoder, path, *, source=False):
    base = args[0]
    tr, h, z, beta = guard._track(signal, args, source=source)
    q, key, initial, target = teacher._mature(tr, args[2][2], args[2][3])
    real, _ = guard._bases(base, signal, q, initial, target, decoder)
    def tensor(a):
        return torch.as_tensor(a, device=beta.device)
    phase = -lag.RADIUS*guard.signed_direction(*real, tensor(key), path)
    h, z, prior = (tensor(a) for a in (h, z, initial))
    error = (tensor(target)-lag._predict(h[q], prior, decoder)).mean(1).cpu().numpy()
    correction, _ = bias.past_bias(q, error)
    correction = tensor(correction)*tensor(args[2][3])
    # Slot zero represents no released forecast, including a fresh shot.
    phase, correction = (torch.cat((a.new_zeros((1, *a.shape[1:])), a)) for a in (phase, correction))
    return q, h, z, beta, prior, phase, correction


def predict_shot(signal, args, decoder, path, archived, *, transport=False, ramp=False):
    base = args[0]
    q, h, z, beta, prior, phase, correction = parent_track(signal, args, decoder, path)
    eligible, fields, target_radius = _feedback_data(signal, q, args[2], beta.device)
    states = _estimate(base, signal, q, eligible, phase, correction, prior, fields, target_radius, decoder,
                       transport=transport)
    curves, dose, aux, parent_err, dose_err = _replay(
        base, signal, h, z, beta, q, eligible, phase, correction, states, archived,
        prior, fields, target_radius, decoder, ramp=ramp)
    return curves, q+lag.H+1, eligible, dose, aux, parent_err, dose_err


def load_reference(lock):
    p, s = (read(REFERENCE/k) for k in ('protocol.json', 'seal.json'))
    signed = Path(p['reference'])
    gp, _, _, args, decoder = bias.eos._load(signed, lock)
    if (p['reference_seal'] != sha(signed/'seal.json') or gp['teacher_sha256'] != sha(signed/'teacher.json')
            or p['source_sha256'] != gp['source_sha256'] or lock['ids'] != p['order']
            or p['calibration'] != gp['calibration'] or p['original'] != gp['original']
            or p['startup'] != gp['startup'] or len(s['curve_sha256']) != 360):
        raise ValueError('invalid frozen reference')
    if args[0].embedding.in_features != args[0].cfg.patch*46:
        raise ValueError('expected separate value and availability lanes')
    path = torch.as_tensor(read(signed/'teacher.json')['path'], device='cuda', dtype=torch.float32)
    return p, s, args, decoder, path


def run(out, *, transport=False, ramp=False):
    lag.configure()
    lock = lag.load_inputs()
    p, s, args, decoder, path = load_reference(lock)
    transport = transport or ramp
    new = RAMP if ramp else TRANSPORT if transport else NEW
    frozen = [state_digest(a) for a in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__), out/'executed_probe.py')
    write(out/'protocol.json', dict(
        scope=('EAST360_RELEASE_HANDOFF_DEVELOPMENT' if ramp else 'EAST360_RELEASE_JACOBIAN_DEVELOPMENT'
               if transport else 'EAST360_SHOT_LOCAL_GROUP_TOKEN_DEVELOPMENT'),
        order=p['order'], calibration=p['calibration'], original=p['original'], startup=p['startup'],
        source_sha256=p['source_sha256'], reference=str(REFERENCE), reference_seal=sha(REFERENCE/'seal.json'),
        executed_sha256=sha(Path(__file__)), arms=(*OLD, *new), horizon=lag.H, stride=lag.STRIDE,
        groups=[[io23.RAW23[c] for c in g] for g in TRUSTED],
        excluded=[io23.RAW23[c] for c in EXCLUDED],
        objective='Huber on five-bin group log radii, using only a known earlier within-shot reference',
        step=('use the released forecast loss covector in the hidden coordinates at issue; write through '
              'the Jacobian of the currently observed release window, or the old issue Jacobian as control; '
              'same-bin rotated group errors use the same release Jacobian' if transport else
              'one shot-local gradient step from zero on a shadow parent forecast; same-bin rotated '
              'group errors and a non-gradient direct write are controls'),
        clock='issue using the latest already released parent state; train only at q+26 or later; '
              'replace temporary state on the next release and discard at shot end; never read outcome',
        write='equal-and-opposite adjacent token changes before block one, common dose at or below '
              'the already present input-embedding displacement; no CLS write',
        handoff=('interpolate previous and current token changes over one source patch from q+26; '
                 'match the smaller realized nominal blend norm across arms, discard prior state if new '
                 'feedback window is ineligible' if ramp else 'replace on release')))
    started = time.monotonic()
    hashes, witnesses, errors = {}, {}, []
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome unavailable before sealing')):
        for rank, sid in enumerate(p['order'], 1):
            name = f'{rank:04d}_{sid}.npz'
            f = REFERENCE/'shots'/name
            if sha(f) != s['curve_sha256'][name] or sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
                raise ValueError('frozen curve or raw data changed')
            with np.load(f, allow_pickle=False) as old:
                t, archived, expected = old['times'], old[bias.BIAS], old['release']
            if rank <= p['startup']:
                curves = np.broadcast_to(archived, (len(new), len(t)))
                release, eligible = np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
                dose, aux = np.zeros(len(t), np.float32), np.empty((0, len(new)+1), np.float32)
            else:
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'], t)
                signal = lag.prepare_actual(raw, rank, lock['mapping'])
                curves, release, eligible, dose, aux, e, d = predict_shot(
                    signal, args, decoder, path, archived, transport=transport, ramp=ramp)
                np.testing.assert_array_equal(release, expected)
                errors.append((e, d))
                if rank in (21, 120, 360):
                    end = min(len(t), 390)
                    short = {k: signal[k][:end] for k in ('values', 'available')}
                    v, r, ok, g, *_ = predict_shot(short, args, decoder, path, archived[:end],
                                                  transport=transport, ramp=ramp)
                    np.testing.assert_array_equal(r, release[release<end])
                    np.testing.assert_array_equal(ok, eligible[release<end])
                    np.testing.assert_allclose(v, curves[:, :end], rtol=0., atol=3e-6)
                    np.testing.assert_allclose(g, dose[:end], rtol=0., atol=2e-6)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(v>=tau, curves[:, :end]>=tau)
                    witnesses[str(sid)] = end
            f = out/'shots'/name
            with f.open('xb') as handle:
                np.savez_compressed(handle, release=release, eligible=eligible, dose=dose, auxiliary=aux,
                                    **dict(zip(new, curves, strict=True)))
            hashes[name] = sha(f)
            if rank == 21 or rank%60 == 0:
                print(dict(stage='group_token_seal', rank=rank, seconds=round(time.monotonic()-started, 1)), flush=True)
    if frozen != [state_digest(a) for a in args[:2]]:
        raise ValueError('frozen model changed')
    parent_error, dose_error = np.max(errors, axis=0)
    write(out/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses,
        max_parent_error=parent_error, max_token_norm_error=dose_error, seconds=time.monotonic()-started))


def score(out):
    p, s = (read(out/k) for k in ('protocol.json', 'seal.json'))
    if (p['reference_seal'] != sha(REFERENCE/'seal.json') or len(s['curve_sha256']) != 360
            or p['executed_sha256'] != sha(out/'executed_probe.py')):
        raise ValueError('incomplete or changed prediction seal')
    ref = read(REFERENCE/'seal.json')['curve_sha256']
    legacy = read(REFERENCE/'per_shot.json')
    arms = tuple(p['arms'])
    if arms == ARMS:
        new, pairs = NEW, PAIRS
    elif arms in ((*OLD, *TRANSPORT), (*OLD, *RAMP)):
        new = arms[len(OLD):]
        pairs = tuple((bias.BIAS, k) for k in new) + ((new[0], new[1]), (new[2], new[1]))
    else:
        raise ValueError('unknown experimental arms')
    rows = {b: {k: [] for k in arms} for b in p['calibration']}
    aux, stats = [], np.zeros(5, dtype=int)
    for rank, sid in enumerate(p['order'], 1):
        name = f'{rank:04d}_{sid}.npz'
        a, b = (root/'shots'/name for root in (out, REFERENCE))
        if sha(a) != s['curve_sha256'][name] or sha(b) != ref[name]:
            raise ValueError('changed sealed curves')
        disrupt, down, t = hdf._raw_truth(sid)
        with np.load(a, allow_pickle=False) as trace, np.load(b, allow_pickle=False) as old:
            np.testing.assert_array_equal(t, old['times'])
            q, eligible, dose = (trace[k] for k in ('release', 'eligible', 'dose'))
            np.testing.assert_array_equal(q, old['release'])
            legal = np.r_[False, eligible][np.searchsorted(q, np.arange(len(t)), side='right')]
            applied = dose > 0.
            if (applied & ~legal).any() or (rank <= p['startup'] and applied.any()):
                raise ValueError('feedback used before release or in the preparation segment')
            late_opportunity = False
            if disrupt and not legacy['0.2'][bias.BIAS][rank-1]['timely']:
                lead = np.round((down-t)*1000., 9)
                late_opportunity = bool(np.any(applied & (lead > 30.) & (lead <= 125.)))
            stats += [len(q), int(eligible.sum()), int(applied.sum()), int(applied.any()), int(late_opportunity)]
            aux.append(trace['auxiliary'])
            curves = {k: (old[k] if k in OLD else trace[k]) for k in arms}
            for k in new:
                np.testing.assert_array_equal(curves[k][~applied], curves[bias.BIAS][~applied])
            for band, tau in p['calibration'].items():
                for k, value in curves.items():
                    row = dict(shot=sid, disruptive=disrupt, down_s=down, **score_curve(
                        t, value, disrupt, down, p['original'] if rank <= p['startup'] else tau))
                    if k in OLD and row != legacy[band][k][rank-1]:
                        raise ValueError('raw HDF5 outcome changed archived scores')
                    rows[band][k].append(row)
    write(out/'per_shot.json', rows)
    write(out/'lead_sensitivity.json', report([out/'per_shot.json'], arms))
    metrics = {k: {n: timing_decomposition(x) for n, x in v.items()} for k, v in rows.items()}
    changes = {k: {f'{a}__to__{b}': flips(v[a], v[b]) for a, b in pairs} for k, v in rows.items()}
    aux = np.concatenate(aux)
    write(out/'result.json', dict(status='COMPLETED_SHOT_LOCAL_GROUP_TOKEN', metrics=metrics, paired=changes,
        raw_verified=360*len(arms)*len(metrics), activity=dict(zip(
            ('mature', 'eligible', 'matched_endpoints', 'matched_shots', 'late_opportunity'),
            stats.tolist(), strict=True)),
        auxiliary=dict(windows=len(aux), mean_huber=dict(zip((bias.BIAS, *new),
                       aux.mean(0).tolist(), strict=True)) if len(aux) else None)))
    print({k: {n: (x['tp'], x['fp']) for n, x in v.items()} for k, v in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--transport', action='store_true', help='recompute the write Jacobian at feedback release')
    parser.add_argument('--ramp', action='store_true', help='blend transported states over one input patch')
    a = parser.parse_args()
    run(a.out.resolve(), transport=a.transport, ramp=a.ramp)
    score(a.out.resolve())
