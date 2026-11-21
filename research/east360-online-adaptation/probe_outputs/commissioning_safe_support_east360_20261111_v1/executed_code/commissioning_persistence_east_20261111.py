"""Full, paired EAST360 replay of a persistent within-shot innovation."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_persistence_20261111 as persistent
from . import commissioning_persistence_learning_20261111 as train
from . import io23, simple_data
from .commissioning_intra_20261110 import STARTUP
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


MODES = ('fixed', 'inherited', *persistent.VIEWS)


@torch.no_grad()
def encode(base, net, signal, params, old_mapper, quiet, mapper, coef):
    x, _ = features(base, signal, 'cuda')
    tr, pre, ready = train.observed_track(net, signal, np.arange(len(x)), params, old_mapper)
    additions, audit = persistent.replay(*ready, quiet)
    key = torch.from_numpy(tr['key'])
    hidden = {'fixed': torch.from_numpy(tr['pre']).tanh().numpy(), 'inherited': pre.tanh().numpy()}
    for name, v in additions.items():
        hidden[name] = train.view(pre, key, torch.from_numpy(v), mapper).numpy()
    d = torch.einsum('...k,...f,kfw->...w', key, torch.from_numpy(additions['sustained']), mapper)
    return scores(x, hidden, coef, MODES), d.norm(dim=1).numpy(), audit


def run(previous, out):
    configure()
    start = time.monotonic()
    contract = load_inputs()
    ids, roles = contract['ids'], io23.source_roles()
    p, previous_seal = (read(previous/k) for k in ('protocol.json', 'seal.json'))
    if (p['scope'] != 'EAST360_FIXED_HEAD_MATCHED_IN_SHOT_OPERATING_CONTEXT_DEVELOPMENT'
            or p['order'] != ids or p['startup'] != STARTUP or p['source_sha256'] != PARENT_SHA256
            or p['fit_source_ids'] != roles['train'] or len(previous_seal['curve_sha256']) != 360
            or any(sha(previous/k) != v for k, v in previous_seal['fit_assets'].items())
            or any(sha(Path(__file__).parent/k) != v for k, v in p['code_sha256'].items())):
        raise ValueError('sealed prior 138/31 run is required')
    ref = Path(p['reference'])
    if any(sha(ref/k) != v for k, v in p['reference_assets'].items()):
        raise ValueError('shared risk head or thresholds changed')
    source = Path(read(ref/'protocol.json')['source_prior'])
    if any(sha(source/k) != v for k, v in p['source_assets'].items()):
        raise ValueError('shared source representation changed')
    forecasts = Path(p['forecast_reference'])/'fit.npz'
    if sha(forecasts) != p['forecast_source_fit_sha256']:
        raise ValueError('shared direct forecaster changed')
    with np.load(forecasts, allow_pickle=False) as f:
        decoder, noise = (f['decoder_weight'], f['decoder_bias']), f['scale']
    with np.load(previous/'mapper.npz', allow_pickle=False) as f:
        old_mapper, context_scale = torch.from_numpy(f['mapper']), f['context_scale']
    info = read(source/'result.json')
    net = LagState(info['inputs'], info['edges']).eval().requires_grad_(False)
    with np.load(source/'model.npz', allow_pickle=False) as f:
        net.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    base, ckpt = load_parent(DEFAULT_PARENT)
    params = (base.center.detach().numpy().copy(), base.scale.detach().numpy().copy(),
              decoder, noise, context_scale)
    base.cuda().requires_grad_(False)
    frozen, shape_frozen = state_digest(base), state_digest(net)
    if frozen != contract['baseline_protocol']['source_state_sha256'] or ckpt['threshold'] != p['original']:
        raise ValueError('qualified source checkpoint changed')
    with np.load(ref/'readout.npz', allow_pickle=False) as f:
        coef = torch.from_numpy(f['coef']).cuda()
    beta = coef[-25:-1]
    original, cutoffs = p['original'], p['calibration']
    advanced = io23.advanced_times()
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    code = (Path(__file__), Path(persistent.__file__), Path(train.__file__))
    write(out/'protocol.json', dict(
        scope='EAST360_PERSISTENT_SIGNED_DEVIATION_FIXED_HEAD_DEVELOPMENT', arms=MODES,
        order=ids, startup=STARTUP, source_sha256=PARENT_SHA256, calibration=cutoffs, original=original,
        previous=str(previous), previous_assets={k: sha(previous/k) for k in (
            'protocol.json','seal.json','mapper.npz','per_shot.json')},
        code_sha256={**p['code_sha256'], **{f.name: sha(f) for f in code}},
        data_bindings=contract['bound'], source_fit_ids=roles['train'], source_held_ids=roles['val'],
        cold_fit_ids=ids[:STARTUP], no_suffix_fit=True, persistent_cross_shot_updates=0,
        reference=dict(history=persistent.HISTORY, neighbors=persistent.NEIGHBORS,
                       excluded_recent=2, robust_prior_strength=persistent.PRIOR_STRENGTH,
                       prior='safe J-TEXT TRAIN only, median and scaled MAD'),
        controls='same-norm current, older-by-one agreement, signed inverse, later-by-one release',
        schedule=dict(first_effect=persistent.FIRST_EFFECT, stride=persistent.STRIDE, window=persistent.WINDOW),
        fit=dict(steps=train.STEPS, seed=train.SEED, map_radius=train.RADIUS, lr=.012),
        inherited='exact prior no-context feedback, no new threshold or persistent target update'))
    source_rows = train.source(net, roles['train'], advanced, params, old_mapper, 'source_train')
    quiet, quiet_windows = train.fit_quiet(source_rows)
    match_scale = np.asarray(info['match_scale'])
    pairs = train.pack(source_rows, quiet, match_scale)
    del source_rows
    cold = []
    for sid in ids[:STARTUP]:
        if sha(io23.path_for('EAST', sid)) != contract['hashes'][str(sid)]:
            raise ValueError('cold-shot evidence changed')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        signal = prepare_actual(raw, STARTUP+1, contract['mapping'])
        cold.append(train.labelled(net, raw, signal, None, params, old_mapper, base=base, coef=coef))
    cold = train.pack(cold, quiet, match_scale)
    print(dict(stage='persistence_pairs', source=len(pairs[1][0]), cold=len(cold[1][0]),
               quiet_windows=quiet_windows), flush=True)
    mapper = train.fit(pairs, cold, beta, cutoffs['0.2'])
    held = train.pack(train.source(net, roles['val'], advanced, params, old_mapper, 'source_held'),
                      quiet, match_scale)
    write(out/'fit.json', dict(
        source_train=train.concordance(pairs, beta.cpu(), mapper),
        source_held=train.concordance(held, beta.cpu(), mapper),
        cold_train=train.concordance(cold, beta.cpu(), mapper),
        quiet_windows=quiet_windows, map_norm=float(mapper.norm()), readout_norm=float(beta.norm())))
    with (out/'state.npz').open('xb') as f:
        np.savez(f, mapper=mapper.numpy(), location=quiet[0], scale=quiet[1])
    fitted = {k: sha(out/k) for k in ('fit.json', 'state.npz')}
    receipts, witness, checks = {}, {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for rank, sid in enumerate(ids, 1):
            name = f'{rank:04d}_{sid}.npz'
            old = previous/'shots'/name
            if sha(old) != previous_seal['curve_sha256'][name]:
                raise ValueError('prior curve changed')
            with np.load(old, allow_pickle=False) as f:
                t, fixed, inherited = f['times'], f['fixed'], f['uncalibrated']
            if rank <= STARTUP:
                curves = dict.fromkeys(MODES, fixed)
                n = np.zeros_like(fixed)
            else:
                if sha(io23.path_for('EAST', sid)) != contract['hashes'][str(sid)]:
                    raise ValueError('EAST signal changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'], t)
                signal = prepare_actual(raw, rank, contract['mapping'])
                args = (params, old_mapper, quiet, mapper, coef)
                curves, n, checks[str(sid)] = encode(base, net, signal, *args)
                np.testing.assert_array_equal(curves['fixed'], fixed)
                np.testing.assert_array_equal(curves['inherited'], inherited)
                for k in persistent.VIEWS:
                    early = persistent.FIRST_EFFECT + (persistent.STRIDE if k == 'lagged' else 0)
                    np.testing.assert_array_equal(curves[k][:early], inherited[:early])
                if rank in (21, 120, 360):
                    witness[str(sid)] = []
                    for length in (persistent.FIRST_EFFECT+4, persistent.FIRST_EFFECT+2*persistent.STRIDE+3):
                        small = {k: signal[k][:length] for k in ('values','available')}
                        pred, radius, _ = encode(base, net, small, *args)
                        np.testing.assert_allclose(radius, n[:len(radius)], rtol=0., atol=2e-6)
                        for k, v in pred.items():
                            real = curves[k][:len(v)]
                            np.testing.assert_allclose(v, real, rtol=0., atol=2e-6)
                            for cutoff in cutoffs.values():
                                np.testing.assert_array_equal(v>=cutoff, real>=cutoff)
                        witness[str(sid)].append(length)
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f, times=t, perturbation_norm=n, **curves)
            receipts[name] = sha(path)
            if rank%60 == 0:
                print(dict(stage='sealed', shots=rank, seconds=time.monotonic()-start), flush=True)
    if (state_digest(base) != frozen or state_digest(net) != shape_frozen
            or any(sha(out/k) != v for k, v in fitted.items())):
        raise ValueError('frozen model changed during replay')
    write(out/'seal.json', dict(curve_sha256=receipts, prefix_witnesses=witness, fitted=fitted))
    write(out/'dynamics.json', checks)
    per = {c: {k: [] for k in MODES} for c in cutoffs}
    for rank, sid in enumerate(ids, 1):
        name = f'{rank:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path) != receipts[name]:
            raise ValueError('unsealed probability track')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as f:
            for c, v in per.items():
                cutoff = original if rank <= STARTUP else cutoffs[c]
                for k in MODES:
                    a = score_curve(f['times'], f[k], d, down, cutoff)
                    v[k].append(dict(shot=sid, disruptive=d, down_s=down, **a))
    past = read(previous/'per_shot.json')
    if any(v['fixed'] != past[c]['fixed'] or v['inherited'] != past[c]['uncalibrated'] for c, v in per.items()):
        raise ValueError('same parent and alarm protocol required')
    metrics = {c: {k: timing_decomposition(v) for k, v in g.items()} for c, g in per.items()}
    write(out/'per_shot.json', per)
    write(out/'result.json', dict(status='COMPLETED_EAST360_SUSTAINED_PAIRED_DEVELOPMENT',
        metrics=metrics,
        paired_vs_inherited={c: {k: flips(v['inherited'],v[k]) for k in persistent.VIEWS}
                             for c, v in per.items()},
        sustained_vs_control={c: {k: flips(v[k],v['sustained']) for k in persistent.VIEWS[1:]}
                              for c, v in per.items()},
        comparisons={k: sum(r[k] for r in checks.values()) for k in next(iter(checks.values()))},
        seconds=time.monotonic()-start))
    print({c: {k: (v['tp'],v['fp']) for k,v in g.items()} for c,g in metrics.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    run(args.previous.resolve(), args.output.resolve())
