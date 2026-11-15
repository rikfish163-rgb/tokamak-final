"""Retest the same fitted in-shot feedback with causal timing and sign controls."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_operating_east_20261111 as prior
from . import io23, simple_data
from .commissioning_intra_20261110 import STARTUP
from .commissioning_lag_state_20261110 import LagState
from .commissioning_split_learning_20261111 import LATE_EFFECT, STRIDE, feature_view, track
from .commissioning_transfer_20261108 import features
from .commissioning_transition_east_20261110 import scores
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


NAMES = ('fixed', 'real', 'lagged', 'sign_flip', 'slope_flip')


def controls(state):
    """Delay by one complete issuance stride or invert signed state coordinates."""
    lag = np.zeros_like(state)
    lag[STRIDE:] = state[:-STRIDE]
    slope = state.copy()
    slope[:, 1::2] *= -1
    return dict(real=state, lagged=lag, sign_flip=-state, slope_flip=slope)


@torch.no_grad()
def encode(base, net, signal, params, mapper, coef):
    x, _ = features(base, signal, 'cuda')
    tr = track(net, signal, *params[:2], np.arange(len(x)))
    states, _ = prior.estimate(tr, signal, *params[2:])
    pre, key = torch.from_numpy(tr['pre']), torch.from_numpy(tr['key'])
    views = {'fixed': pre.tanh().numpy(), **{
        k: feature_view(pre, key, torch.from_numpy(v), mapper).numpy()
        for k, v in controls(states['uncalibrated']).items()}}
    return scores(x, views, coef, NAMES)


def run(previous, out):
    configure()
    start = time.monotonic()
    data = load_inputs()
    p, s = (read(previous/k) for k in ('protocol.json', 'seal.json'))
    if (p['scope'] != 'EAST360_FIXED_HEAD_MATCHED_IN_SHOT_OPERATING_CONTEXT_DEVELOPMENT'
            or p['order'] != data['ids'] or p['startup'] != STARTUP or p['source_sha256'] != PARENT_SHA256
            or any(sha(previous/k) != v for k, v in s['fit_assets'].items())
            or any(sha(Path(__file__).parent/k) != v for k, v in p['code_sha256'].items())):
        raise ValueError('completed operating-context experiment required')
    reference = Path(p['reference'])
    if any(sha(reference/k) != v for k, v in p['reference_assets'].items()):
        raise ValueError('shared static assets changed')
    shape_dir = Path(read(reference/'protocol.json')['source_prior'])
    if any(sha(shape_dir/k) != v for k, v in p['source_assets'].items()):
        raise ValueError('shared representation changed')
    forecast = Path(p['forecast_reference'])/'fit.npz'
    if sha(forecast) != p['forecast_source_fit_sha256']:
        raise ValueError('direct decoder changed')
    with np.load(forecast, allow_pickle=False) as f:
        decoder, noise = (f['decoder_weight'], f['decoder_bias']), f['scale']
    with np.load(previous/'mapper.npz', allow_pickle=False) as f:
        mapper, scale = torch.from_numpy(f['mapper']), f['context_scale']
    info = read(shape_dir/'result.json')
    net = LagState(info['inputs'], info['edges']).eval().requires_grad_(False)
    with np.load(shape_dir/'model.npz', allow_pickle=False) as f:
        net.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    base, ckpt = load_parent(DEFAULT_PARENT)
    params = (base.center.detach().numpy().copy(), base.scale.detach().numpy().copy(),
              decoder, noise, scale)
    base.cuda().requires_grad_(False)
    frozen, shape_frozen = state_digest(base), state_digest(net)
    if frozen != data['baseline_protocol']['source_state_sha256'] or ckpt['threshold'] != p['original']:
        raise ValueError('source model mismatch')
    with np.load(reference/'readout.npz', allow_pickle=False) as f:
        coef = torch.from_numpy(f['coef']).cuda()
    original, cutoffs = p['original'], p['calibration']
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    write(out/'protocol.json', dict(
        scope='EAST360_FROZEN_FIT_TIMING_AND_SIGN_CONTROLS_DEVELOPMENT', arms=NAMES,
        order=p['order'], startup=STARTUP, source_sha256=PARENT_SHA256,
        calibration=cutoffs, original=original, first_effect=LATE_EFFECT, stride=STRIDE,
        previous=str(previous), previous_assets={k: sha(previous/k) for k in
            ('protocol.json', 'seal.json', 'mapper.npz', 'per_shot.json')},
        new_fitting=False, persistent_updates=0, current_outcome_for_prediction=False,
        code_sha256={Path(__file__).name: sha(Path(__file__)), **p['code_sha256']}))
    hashes, witnesses = {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for i, sid in enumerate(p['order'], 1):
            name = f'{i:04d}_{sid}.npz'
            parent = previous/'shots'/name
            if sha(parent) != s['curve_sha256'][name]:
                raise ValueError('sealed prior curve changed')
            with np.load(parent, allow_pickle=False) as f:
                t, fixed, real = f['times'], f['fixed'], f['uncalibrated']
            if i <= STARTUP:
                curves = dict.fromkeys(NAMES, fixed)
            else:
                if sha(io23.path_for('EAST', sid)) != data['hashes'][str(sid)]:
                    raise ValueError('EAST source signal changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'], t)
                signal = prepare_actual(raw, i, data['mapping'])
                curves = encode(base, net, signal, params, mapper, coef)
                np.testing.assert_array_equal(curves['fixed'], fixed)
                np.testing.assert_array_equal(curves['real'], real)
                for k in NAMES[1:]:
                    n = LATE_EFFECT+STRIDE if k == 'lagged' else LATE_EFFECT
                    np.testing.assert_array_equal(curves[k][:n], fixed[:n])
                if i in (21, 120, 360):
                    small = {k: signal[k][:LATE_EFFECT+2*STRIDE+4] for k in ('values', 'available')}
                    preview = encode(base, net, small, params, mapper, coef)
                    for k, z in preview.items():
                        actual = curves[k][:len(z)]
                        np.testing.assert_allclose(z, actual, atol=2e-6, rtol=0)
                        for tau in cutoffs.values():
                            np.testing.assert_array_equal(z>=tau, actual>=tau)
                    witnesses[str(sid)] = len(preview['fixed'])
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f, times=t, **curves)
            hashes[name] = sha(path)
            if i%120 == 0:
                print(dict(stage='sealed_controls', shots=i, seconds=time.monotonic()-start), flush=True)
    if state_digest(base) != frozen or state_digest(net) != shape_frozen:
        raise ValueError('frozen predictor changed during replay')
    write(out/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses))
    per = {c: {k: [] for k in NAMES} for c in cutoffs}
    for i, sid in enumerate(p['order'], 1):
        name = f'{i:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path) != hashes[name]:
            raise ValueError('prediction changed after seal')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as f:
            for c, group in per.items():
                for k in NAMES:
                    a = score_curve(f['times'], f[k], d, down, original if i <= STARTUP else cutoffs[c])
                    group[k].append(dict(shot=sid, disruptive=d, down_s=down, **a))
    old = read(previous/'per_shot.json')
    if any(g['fixed'] != old[c]['fixed'] or g['real'] != old[c]['uncalibrated'] for c, g in per.items()):
        raise ValueError('old frozen or real first alarms do not match')
    metrics = {c: {k: timing_decomposition(rows) for k, rows in g.items()} for c, g in per.items()}
    write(out/'per_shot.json', per)
    write(out/'result.json', dict(status='COMPLETED_FROZEN_FIT_MECHANISM_CONTROL_EAST360_DEVELOPMENT',
        metrics=metrics, paired_vs_fixed={c: {k: flips(g['fixed'], g[k]) for k in NAMES[1:]}
                                        for c, g in per.items()},
        real_vs_controls={c: {k: flips(g[k], g['real']) for k in NAMES[2:]} for c, g in per.items()},
        seconds=time.monotonic()-start))
    print({c: {k: (v['tp'], v['fp']) for k, v in g.items()} for c, g in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    run(a.previous.resolve(), a.output.resolve())
