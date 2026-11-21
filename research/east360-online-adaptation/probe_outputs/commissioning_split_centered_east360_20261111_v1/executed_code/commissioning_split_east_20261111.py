"""Sealed EAST360 replay of an early/late update to a shared direct forecast."""

import argparse
from functools import partial
from inspect import unwrap
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_split_learning_20261111 as learn
from . import commissioning_split_steps_20261111 as steps
from . import commissioning_weak_shape_source_20261111 as weak
from . import io23, simple_data
from .commissioning_intra_20261110 import CAPS, STARTUP
from .commissioning_lag_state_20261110 import LagState
from .commissioning_transfer_20261108 import features
from .commissioning_transition_east_20261110 import scores
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


MODES = ('fixed', 'split', 'delayed', 'reversed')


@torch.no_grad()
def encode(base, net, signal, center, scale, decoder, noise, mapper, coef, replay):
    raw, _ = features(base, signal, 'cuda')
    tr = learn.track(net, signal, center, scale, np.arange(len(raw)))
    start = time.perf_counter()
    prediction = learn.base_forecast(tr, decoder)
    state, audit = replay(prediction, learn.normalized_target(tr, noise),
                          tr['observed'], tr['q'], len(raw))
    state_ms = 1000. * (time.perf_counter() - start)
    other = ~learn.early_phase(tr['ends'])
    np.testing.assert_array_equal(state['split'][other], state['delayed'][other])
    pre, key = torch.from_numpy(tr['pre']), torch.from_numpy(tr['key'])
    views = {'fixed': pre.tanh().numpy()}
    for name, v in state.items():
        views[name] = learn.feature_view(pre, key, torch.from_numpy(v), mapper).numpy()
    result = scores(raw, views, coef, MODES)
    return result, dict(audit, batched_forecast_and_state_ms=state_ms,
                        distinct_early_state_points=int(np.any(state['split'] != state['delayed'], axis=1).sum()))


def run(reference, out, *, centered_slope=False):
    configure()
    started = time.monotonic()
    locked = load_inputs()
    p, seal, cal = (read(reference / k) for k in ('protocol.json', 'seal.json', 'calibration.json'))
    if (p['scope'] != 'EAST360_PAIRED_CONTEXT_MATCHED_WEAK_SHAPE_DEVELOPMENT'
            or p['order'] != locked['ids'] or p['startup'] != STARTUP
            or p['source_sha256'] != PARENT_SHA256 or len(seal['curve_sha256']) != 360
            or any(sha(reference / k) != v for k, v in seal['artifacts'].items())):
        raise ValueError('sealed 136/32 baseline required')
    source = Path(p['source_prior'])
    if (any(sha(source / k) != v for k, v in p['source_assets'].items())
            or read(source / 'protocol.json')['code_sha256'] != weak.code_bindings()):
        raise ValueError('frozen shape producer changed')
    info = read(source / 'result.json')
    net = LagState(info['inputs'], info['edges']).eval().requires_grad_(False)
    with np.load(source / 'model.npz', allow_pickle=False) as f:
        net.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    base, ckpt = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy().copy() for v in (base.center, base.scale))
    base.cuda().requires_grad_(False)
    frozen, frozen_shape = state_digest(base), state_digest(net)
    if frozen != locked['baseline_protocol']['source_state_sha256'] or ckpt['threshold'] != p['original']:
        raise ValueError('qualified source state changed')
    with np.load(reference / 'readout.npz', allow_pickle=False) as f:
        coef = torch.from_numpy(f['coef']).cuda()
    beta = coef[-learn.WIDTH-1:-1]
    replay = partial(steps.replay, centered=centered_slope)
    original, thresholds = p['original'], cal['cutoffs']
    ids, roles, advanced = locked['ids'], io23.source_roles(), io23.advanced_times()
    out.mkdir(parents=True)
    (out / 'shots').mkdir()
    code = (Path(__file__), Path(learn.__file__), Path(steps.__file__),
            Path(unwrap(scores).__code__.co_filename), Path(learn.pairs.__file__),
            Path(learn.fit_decoder.__code__.co_filename))
    write(out / 'protocol.json', dict(
        scope='EAST360_DIRECT25_SPLIT_RELEASE_FIXED_HEAD_DEVELOPMENT', order=ids, startup=STARTUP,
        arms=MODES, original=original, calibration=thresholds, source_sha256=PARENT_SHA256,
        reference=str(reference), reference_assets={k: sha(reference/k) for k in (
            'protocol.json', 'seal.json', 'calibration.json', 'readout.npz')},
        source_assets=p['source_assets'], data_bindings=locked['bound'],
        source_label_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
        code_sha256={f.name: sha(f) for f in code}, fit_source_ids=roles['train'],
        source_forecast_fit='direct 25 by 10 past-conditioned waveform departures; J-TEXT TRAIN only',
        fit_cold_ids=ids[:STARTUP], cold_use='ended outcomes teach adapter; predictions stay archived',
        no_suffix_target_fitting=True, persistent_suffix_updates=0,
        feedback='centered_slope' if centered_slope else 'level_and_slope',
        schedule=dict(first_origin=learn.FIRST, stride=learn.STRIDE, observed_stages=[8, 17],
                      first_effect=learn.EARLY_EFFECT, delayed_first_effect=learn.LATE_EFFECT,
                      delayed='same steps and order released together only when all 25 arrive',
                      reversed=('same issued forecast and observation support; flip signed slope only'
                                if centered_slope else
                                'same segment and mask; reverse chronological values within observed slots')),
        fit=dict(steps=learn.STEPS, batch=learn.BATCH, seed=learn.SEED,
                 source_weight=.65, cold_weight=.35, mapper_radius=learn.MAP_RADIUS,
                 frozen_alarm_and_heads=True)))
    src = learn.collect_source(net, roles['train'], advanced, center, scale, 'train')
    decoder, noise, receipt = learn.fit_forecast(src)
    match_scale = np.asarray(info['match_scale'])
    train = learn.pack(src, decoder, noise, replay, match_scale)
    del src
    cold_data = []
    for sid in ids[:STARTUP]:
        if sha(io23.path_for('EAST', sid)) != locked['hashes'][str(sid)]:
            raise ValueError('cold evidence changed')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        signal = prepare_actual(raw, STARTUP+1, locked['mapping'])
        cold_data.append(learn.episode(net, raw, 'EAST', None, center, scale, signal=signal))
    cold = learn.pack(cold_data, decoder, noise, replay, match_scale)
    print(dict(stage='fit_pairs', source=len(train[1][0]), cold=len(cold[1][0])), flush=True)
    mapper = learn.teach(train, cold, beta)
    val = learn.collect_source(net, roles['val'], advanced, center, scale, 'val')
    held = learn.pack(val, decoder, noise, replay, match_scale)
    del val
    fitted = dict(receipt, mapper_norm=float(mapper.norm()),
                  source=learn.pair_receipt(*train[:2]), cold=learn.pair_receipt(*cold[:2]))
    validate = dict(source_val=learn.pair_receipt(*held[:2]),
                    source_val_order=learn.check_pairs(*held[:2], mapper, beta.cpu()),
                    cold_training_order=learn.check_pairs(*cold[:2], mapper, beta.cpu()),
                    forecast_columns=['observed_values', 'ridge_squared_error', 'persistence_squared_error'],
                    forecast_rows=['short', 'tail'], source_val_forecast=held[2].tolist(),
                    cold_forecast=cold[2].tolist())
    write(out / 'fit.json', fitted)
    write(out / 'source_validation.json', validate)
    with (out / 'fit.npz').open('xb') as f:
        np.savez(f, mapper=mapper.numpy(), decoder_weight=decoder[0], decoder_bias=decoder[1], scale=noise)
    fit_assets = {k: sha(out / k) for k in ('fit.json', 'fit.npz', 'source_validation.json')}
    print(dict(stage='source_val_pairs', report=validate['source_val_order']), flush=True)
    hashes, witness, dynamics = {}, {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for i, sid in enumerate(ids, 1):
            name = f'{i:04d}_{sid}.npz'
            archive = reference / 'shots' / name
            if sha(archive) != seal['curve_sha256'][name]:
                raise ValueError('sealed baseline curve changed')
            with np.load(archive, allow_pickle=False) as f:
                t, fixed = f['times'], f['fixed']
            if i <= STARTUP:
                curves = dict.fromkeys(MODES, fixed)
            else:
                if sha(io23.path_for('EAST', sid)) != locked['hashes'][str(sid)]:
                    raise ValueError('raw EAST signal changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'], t)
                signal = prepare_actual(raw, i, locked['mapping'])
                args = (center, scale, decoder, noise, mapper, coef, replay)
                curves, dynamics[str(sid)] = encode(base, net, signal, *args)
                np.testing.assert_array_equal(curves['fixed'], fixed)
                for k in MODES[1:]:
                    n = learn.LATE_EFFECT if k == 'delayed' else learn.EARLY_EFFECT
                    np.testing.assert_array_equal(curves[k][:n], fixed[:n])
                if i in (21, 120, 360):
                    witness[str(sid)] = {}
                    for length in (learn.EARLY_EFFECT+4, learn.LATE_EFFECT+90):
                        small = {k: signal[k][:length] for k in ('values', 'available')}
                        preview, _ = encode(base, net, small, *args)
                        for k, s in preview.items():
                            full = curves[k][:len(s)]
                            np.testing.assert_allclose(s, full, atol=2e-6, rtol=0)
                            for tau in thresholds.values():
                                np.testing.assert_array_equal(s >= tau, full >= tau)
                        witness[str(sid)][str(length)] = {
                            k: float(np.abs(v - curves[k][:len(v)]).max()) for k, v in preview.items()}
            path = out / 'shots' / name
            with path.open('xb') as f:
                np.savez_compressed(f, times=t, **curves)
            hashes[name] = sha(path)
            if i % 60 == 0:
                print(dict(stage='sealed', shots=i, seconds=time.monotonic()-started), flush=True)
    if (state_digest(base) != frozen or state_digest(net) != frozen_shape
            or any(sha(out / k) != v for k, v in fit_assets.items())):
        raise ValueError('frozen parameters changed during replay')
    write(out / 'seal.json', dict(curve_sha256=hashes, fit_assets=fit_assets, prefix_witnesses=witness))
    write(out / 'dynamics_by_shot.json', dynamics)
    per = {c: {k: [] for k in MODES} for c in CAPS}
    for i, sid in enumerate(ids, 1):
        name = f'{i:04d}_{sid}.npz'
        path = out / 'shots' / name
        if sha(path) != hashes[name]:
            raise ValueError('post-seal curve changed')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as f:
            for c, group in per.items():
                for k in MODES:
                    a = score_curve(f['times'], f[k], d, down, original if i <= STARTUP else thresholds[c])
                    group[k].append(dict(shot=sid, disruptive=d, down_s=down, **a))
    archived = read(reference / 'per_shot.json')
    if any(g['fixed'] != archived[c]['fixed'] for c, g in per.items()):
        raise ValueError('archived fixed per-shot results differ')
    metrics = {c: {k: timing_decomposition(r) for k, r in g.items()} for c, g in per.items()}
    write(out / 'per_shot.json', per)
    write(out / 'result.json', dict(
        status='COMPLETED_DIRECT25_SPLIT_RELEASE_EAST360_DEVELOPMENT', metrics=metrics,
        paired_vs_fixed={c: {k: flips(g['fixed'], g[k]) for k in MODES[1:]} for c, g in per.items()},
        early_feedback_comparisons={c: {k: flips(g[k], g['split']) for k in ('delayed', 'reversed')}
                                    for c, g in per.items()},
        visible_updates={k: sum(r['visible_updates'][k] for r in dynamics.values()) for k in MODES[1:]},
        distinct_early_state_points=sum(r['distinct_early_state_points'] for r in dynamics.values()),
        seconds=time.monotonic()-started))
    print({c: {k: (v['tp'], v['fp']) for k, v in g.items()} for c, g in metrics.items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--centered-slope', action='store_true')
    a = cli.parse_args()
    run(a.reference.resolve(), a.output.resolve(), centered_slope=a.centered_slope)
