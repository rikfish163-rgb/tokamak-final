"""Identify temporary gains shared by direct waveform forecasts and the fixed risk readout."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.linalg import solve
import torch

from . import commissioning_persistence_20261111 as persistent
from . import commissioning_persistence_learning_20261111 as history
from . import commissioning_radial_east_20261111 as parent
from . import commissioning_split_learning_20261111 as direct
from . import io23, simple_data
from .commissioning_innovation_east_20261111 import fit_decoder
from .commissioning_transfer_20261108 import features
from .commissioning_transition_east_20261110 import scores
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .latent_features import extract_shot
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


BASES, STARTUP = parent.BASES, parent.STARTUP
TRUE, REVERSED, BOTH = 'shared_true', 'shared_reversed', 'shared_true_plus_legacy'
ARMS = (*BASES, TRUE, REVERSED, BOTH)
COMPARISONS = (('fixed', TRUE), ('inherited', TRUE), (REVERSED, TRUE),
               (BASES[-1], TRUE), (BASES[-1], BOTH), (TRUE, BOTH))
RADIUS, RIDGE_FRACTION = .35, .25
HORIZON, STRIDE = direct.HORIZON, direct.STRIDE
FIRST_EFFECT = direct.FIRST + HORIZON + 1


def gain_delta(h, theta):
    return np.tanh(h) * theta


def fit_window(jac, residual, *, radius=RADIUS, prior=None):
    """A Huber-weighted, ridge-regularized local solve in forecast-observed directions."""
    j, r = jac.astype(np.float64), residual.astype(np.float64)
    inv = 1. / np.maximum(1., np.abs(r))
    gram = j.T @ (j * inv[:, None])
    curvature = float(gram.trace() / len(gram))
    if curvature == 0.:
        return np.zeros(j.shape[1], dtype=np.float32)
    target = r if prior is None else r + j @ prior
    gram.flat[::len(gram) + 1] += RIDGE_FRACTION * curvature
    candidate = solve(gram, j.T @ (inv * target), assume_a='pos')
    return (candidate * (radius / max(radius, np.linalg.norm(candidate)))).astype(np.float32)


def _reverse_innovations(innovation, mask):
    flipped = np.zeros(mask.shape, dtype=innovation.dtype)
    flipped[mask] = innovation
    for c in range(mask.shape[1]):
        m = mask[:, c]
        flipped[m, c] = flipped[m, c][::-1]
    return flipped[mask]


def _loss(r):
    a = np.abs(r.astype(np.float64))
    small = np.minimum(a, 1.)
    return np.sum(.5 * small**2 + a - small)


def replay(h, q, weights, forecast, target, mask):
    """Issue a whole window, then estimate gains from its arrived measurements.

    Every block starts from an immutable zero-gain forecast plus the last
    available gain. There is no label, update before q+26, or carry to a new
    shot. The control swaps the order of observed errors within each sensor
    and has the same gain norm and use interval as the true arm.
    """
    n, dim = h.shape
    gains = np.zeros((2, n, dim), np.float32)
    issued = np.stack((forecast.copy(), forecast.copy()))
    mass = np.zeros((len(q), 4), np.float64)
    norms = np.zeros((len(q), 2), np.float32)
    influence = np.zeros((len(q), 2), np.float32)
    releases, before_fit, accepted_fit = [], [], []
    for i, origin in enumerate(q):
        issued[:, i] += (gain_delta(h[origin], gains[:, origin]) @ weights.T).reshape(
            2, *forecast.shape[1:])
        release = int(origin + HORIZON + 1)
        if release >= n:
            break
        m = mask[i]
        if not m.any():
            continue
        sel = weights[m.ravel()] * np.tanh(h[origin])
        y = target[i][m]
        residual = y - issued[0, i][m]
        start_gain = gains[0, origin]
        fitted = fit_window(sel, residual, prior=start_gain)
        reverse = fit_window(sel, _reverse_innovations(residual, m), prior=start_gain)
        norm, rev_norm = float(np.linalg.norm(fitted)), float(np.linalg.norm(reverse))
        if rev_norm == 0. and norm != 0.:
            # Keep the paired experiment on common directional support.
            fitted = np.zeros_like(fitted)
            norm = 0.
        if rev_norm > 0.:
            reverse *= norm / rev_norm
        gains[:, release:release + STRIDE] = np.stack((fitted, reverse))[:, None, :]
        releases.append(release)
        norms[i] = np.linalg.norm(gains[:, release], axis=1)
        mass[i] = m.sum(), _loss(y - forecast[i][m]), _loss(residual), _loss(y - issued[1, i][m])
        before_fit.append(_loss(residual))
        accepted_fit.append(_loss(y - forecast[i][m] - sel @ fitted))
        influence[i] = np.linalg.norm(sel @ fitted), np.linalg.norm(sel @ reverse)
    return gains, dict(release=np.asarray(releases, dtype=np.int64), norm=norms,
                       forecast_huber_sum=mass, forecast_change=influence,
                       issued_true=issued[0], issued_reversed=issued[1],
                       before_fit_error=np.asarray(before_fit), accepted_fit_error=np.asarray(accepted_fit))


def source_fit(args):
    base, net, (center, scale, forecast, noise, _), *_ = args
    ids, advanced = io23.source_roles()['train'], io23.advanced_times()
    latent, target, avail, used = [], [], [], []
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        if len(raw['times']) >= FIRST_EFFECT:
            truth = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
            # Match the structural absence of MNM, not outcomes or suffix statistics.
            raw['available'][:, io23.RAW23.index('MNM')] = False
            q = np.arange(direct.FIRST, len(raw['times']), STRIDE)
            tr = direct.track(net, raw, center, scale, q)
            end = q + HORIZON
            mask = tr['observed']
            keep = (end < len(truth['valid'])) & truth['valid'][np.minimum(end, len(truth['valid']) - 1)]
            keep &= mask.any((1, 2))
            if keep.any():
                h, _ = extract_shot(base, raw, 'cuda')
                diff = direct.normalized_target(tr, noise) - direct.base_forecast(tr, forecast)
                latent.append(h[q[keep]])
                target.append(diff[keep].reshape(np.count_nonzero(keep), -1))
                avail.append(mask[keep].reshape(np.count_nonzero(keep), -1))
                used.append(int(sid))
        if i % 250 == 0 or i == len(ids):
            print(dict(stage='fit_source_forecast', shots=i, total=len(ids)), flush=True)
    h, y, m = map(np.concatenate, (latent, target, avail))
    center = h.mean(0)
    scale = h.std(0)
    if not np.isfinite(scale).all() or (scale <= 0.).any():
        raise ValueError('nonconstant source risk features required')
    w, b = fit_decoder((h - center) / scale, y, m, ridge=.03)
    info = dict(used=used, allowed=ids, origins=len(h), observed=int(m.sum()),
                source_label_scope='only excludes windows after source HDF5 DownTime')
    return (center, scale, w, b), info


@torch.no_grad()
def encode(signal, args, fit):
    base, net, params, old, quiet, mapper, coef, _ = args
    center, scale, w, b = fit
    n = len(signal['values'])
    tr, inherited, ready = history.observed_track(net, signal, np.arange(n), params, old)
    states, _ = persistent.replay(*ready, quiet)
    key = torch.from_numpy(tr['key'])
    latest = inherited + torch.einsum('...k,...f,kfw->...w', key,
                4.*torch.from_numpy(states['current_equal_norm']), mapper)
    x, _ = features(base, signal, 'cuda')
    width = len(center)
    h = (x[:, :width] - center) / scale
    q = tr['q']
    f = direct.base_forecast(tr, params[2]) + (h[q] @ w.T + b).reshape(len(q), HORIZON, 10)
    state, tele = replay(h, q, w, f, direct.normalized_target(tr, params[3]), tr['observed'])
    views = {k: v.tanh().numpy() for k, v in zip(
        BASES, (torch.from_numpy(tr['pre']), inherited, latest), strict=True)}
    curves = scores(x, views, coef, BASES)
    for i, name, view in ((0, TRUE, 'inherited'), (1, REVERSED, 'inherited'), (0, BOTH, BASES[-1])):
        adapted = x.copy()
        adapted[:, :width] += gain_delta(h, state[i]) * scale
        curves.update(scores(adapted, {name: views[view]}, coef, (name,)))
        zero = ~state[i].any(1)
        np.testing.assert_array_equal(curves[name][zero], curves[view][zero])
    np.testing.assert_allclose(tele['norm'][:, 0], tele['norm'][:, 1], atol=1e-7, rtol=2e-6)
    if not all(np.isfinite(y).all() for y in curves.values()):
        raise FloatingPointError('nonfinite sealed alarm scores')
    valid = q + HORIZON + 1 < n
    keep = ('release', 'norm', 'forecast_huber_sum', 'forecast_change',
            'before_fit_error', 'accepted_fit_error')
    trace = {k: tele[k][valid] if k in ('norm', 'forecast_huber_sum', 'forecast_change') else tele[k]
             for k in keep}
    trace['changed_points'] = np.count_nonzero(state.any(2), axis=1)
    return curves, trace


def run(reference, coordinates, output):
    configure()
    started = time.monotonic()
    lock = load_inputs()
    ids = lock['ids']
    p, old_seal, args, assets = parent._frozen(reference, coordinates, lock)
    frozen = tuple(state_digest(m) for m in args[:2])
    original, cutoffs = p['original'], p['calibration']
    producer = Path(__file__)
    code = dict(p['code_sha256'], **{f.name: sha(f) for f in
                (producer, Path(parent.__file__), Path(extract_shot.__code__.co_filename))})
    (output/'shots').mkdir(parents=True)
    archive = output/'executed_code'
    archive.mkdir()
    for name, value in code.items():
        copyfile(producer.parent/name, archive/name)
        if sha(archive/name) != value:
            raise ValueError(f'archived producer changed: {name}')
    fit, source = source_fit(args)
    np.savez(output/'forecast.npz', center=fit[0], scale=fit[1], weight=fit[2], bias=fit[3])
    write(output/'protocol.json', dict(
        scope='EAST360_SHARED_DIRECT_FORECAST_REPRESENTATION_DEVELOPMENT', order=ids,
        startup=STARTUP, reference=str(reference), assets=assets, code_sha256=code,
        source_sha256=p['source_sha256'], arms=ARMS, comparisons=COMPARISONS,
        original=original, calibration=cutoffs, source_fit=source,
        forecast_sha256=sha(output/'forecast.npz'), radius=RADIUS, ridge_fraction=RIDGE_FRACTION,
        first_effect=FIRST_EFFECT, window=HORIZON, stride=STRIDE,
        outcome_access='seal all 360 risk curves before reading any EAST outcome',
        update='latest observed direct forecast errors; risk head and encoder weights frozen',
        order_control='reverse observed signed innovations within each channel, match gain L2',
        persistent_updates=0))
    hashes, witnesses = {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for rank, sid in enumerate(ids, 1):
            name = f'{rank:04d}_{sid}.npz'
            path = reference/'shots'/name
            if sha(path) != old_seal['curve_sha256'][name]:
                raise ValueError('locked reference curve changed')
            with np.load(path, allow_pickle=False) as f:
                times, arch = f['times'], {k: f[k] for k in BASES}
            trace = {}
            if rank <= STARTUP:
                curves = dict.fromkeys(ARMS, arch['fixed'])
            else:
                if sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
                    raise ValueError('locked signal file changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'], times)
                signal = prepare_actual(raw, rank, lock['mapping'])
                curves, trace = encode(signal, args, fit)
                for k in BASES:
                    np.testing.assert_array_equal(curves[k], arch[k])
                if rank in (21, 120, 360):
                    end = FIRST_EFFECT + 3*STRIDE + 5
                    short, a = encode({k: signal[k][:end] for k in ('values', 'available')}, args, fit)
                    for k in ARMS:
                        np.testing.assert_allclose(short[k], curves[k][:end], atol=2e-6, rtol=0.)
                        for tau in cutoffs.values():
                            np.testing.assert_array_equal(short[k] >= tau, curves[k][:end] >= tau)
                    np.testing.assert_array_equal(a['release'], trace['release'][trace['release'] < end])
                    witnesses[str(sid)] = end
            target = output/'shots'/name
            with target.open('xb') as f:
                np.savez_compressed(f, times=times, **curves, **trace)
            hashes[name] = sha(target)
            if rank % 40 == 0 or rank == STARTUP+1:
                print(dict(stage='sealed', shots=rank, seconds=round(time.monotonic()-started, 2)), flush=True)
    if tuple(state_digest(m) for m in args[:2]) != frozen:
        raise ValueError('frozen models changed')
    write(output/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses))
    per = {c: {k: [] for k in ARMS} for c in cutoffs}
    forecast = np.zeros(4)
    for rank, sid in enumerate(ids, 1):
        name = f'{rank:04d}_{sid}.npz'
        path = output/'shots'/name
        if sha(path) != hashes[name]:
            raise ValueError('unsealed curve')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as f:
            if rank > STARTUP:
                forecast += f['forecast_huber_sum'].sum(0)
            for c, group in per.items():
                tau = original if rank <= STARTUP else cutoffs[c]
                for k in ARMS:
                    group[k].append(dict(shot=sid, disruptive=d, down_s=down,
                                         **score_curve(f['times'], f[k], d, down, tau)))
    prior = read(reference/'per_shot.json')
    if any(per[c][k] != prior[c][k] for c in cutoffs for k in BASES):
        raise ValueError('frozen parent scoring changed')
    write(output/'per_shot.json', per)
    metrics = {c: {k: timing_decomposition(v) for k, v in g.items()} for c, g in per.items()}
    paired = {c: {f'{a}__to__{b}': flips(g[a], g[b]) for a, b in COMPARISONS} for c, g in per.items()}
    write(output/'lead_sensitivity.json', report([output/'per_shot.json'], ARMS))
    losses = dict(zip(('zero', TRUE, REVERSED), (forecast[1:]/forecast[0]).tolist(), strict=True))
    write(output/'result.json', dict(status='COMPLETED_EAST360_SHARED_REPRESENTATION_DEVELOPMENT',
        metrics=metrics, paired=paired, forecast_mean_huber=losses,
        forecast_observations=int(forecast[0]), seconds=time.monotonic()-started))
    print(dict(counts={c: {k: (m['tp'], m['fp']) for k, m in g.items()}
                      for c, g in metrics.items()}, forecast_mean_huber=losses), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('reference', 'coordinates', 'output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    a = parser.parse_args()
    run(a.reference.resolve(), a.coordinates.resolve(), a.output.resolve())
