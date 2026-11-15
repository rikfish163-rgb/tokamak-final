"""Compare progressive temporal support and matched within-shot adaptation on EAST."""

import argparse
from inspect import unwrap
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_prefix_learning_20261111 as prefix
from . import commissioning_split_learning_20261111 as split
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


MODES = ('fixed', 'ramp', 'learned', 'split', 'delayed', 'reversed')


@torch.no_grad()
def encode(base, net, layer, signal, center, scale, match_scale, coef, fast):
    raw, _ = features(base, signal, 'cuda')
    q = np.arange(len(raw))
    row = prefix.record(net, signal, center, scale, q, match_scale)
    args = prefix._tensor(row, 'cpu')
    pre = args[0]
    weight = torch.from_numpy(prefix.history_weight(row['samples']))
    change = prefix.correction(layer, args)
    learned = pre + change
    old = pre.clone()
    old[:split.FIRST] = 0.
    views = dict(fixed=old.tanh().numpy(), ramp=prefix.latent(pre, weight, torch.zeros_like(pre)).numpy(),
                 learned=(weight[:, None]*learned.tanh()).numpy())
    tr = split.track(net, signal, center, scale, np.empty(0, dtype=int))
    forecast = split.base_forecast(tr, fast['decoder'])
    states, audit = steps.replay(forecast, split.normalized_target(tr, fast['scale']),
                                 tr['observed'], tr['q'], len(raw), centered=True)
    other = ~split.early_phase(q)
    np.testing.assert_array_equal(states['split'][other], states['delayed'][other])
    np.testing.assert_array_equal(states['split'], -states['reversed'])
    key = torch.from_numpy(row['key'])
    for name, state in states.items():
        h = split.feature_view(learned, key, torch.from_numpy(state), fast['mapper'])
        views[name] = (weight[:, None]*h).numpy()
    return scores(raw, views, coef, MODES), audit


def run(reference, fast_reference, out):
    configure()
    started = time.monotonic()
    locked = load_inputs()
    p, seal, cal = (read(reference/k) for k in ('protocol.json', 'seal.json', 'calibration.json'))
    if (p['scope'] != 'EAST360_PAIRED_CONTEXT_MATCHED_WEAK_SHAPE_DEVELOPMENT'
            or p['order'] != locked['ids'] or p['startup'] != STARTUP
            or p['source_sha256'] != PARENT_SHA256 or len(seal['curve_sha256']) != 360
            or any(sha(reference/k) != v for k, v in seal['artifacts'].items())):
        raise ValueError('sealed 136/32 static reference required')
    fast_p, fast_s = (read(fast_reference/k) for k in ('protocol.json', 'seal.json'))
    if (fast_p['scope'] != 'EAST360_DIRECT25_SPLIT_RELEASE_FIXED_HEAD_DEVELOPMENT'
            or fast_p['feedback'] != 'centered_slope' or Path(fast_p['reference']) != reference
            or fast_p['order'] != p['order'] or fast_p['fit_cold_ids'] != p['order'][:STARTUP]
            or any(sha(fast_reference/k) != v for k, v in fast_s['fit_assets'].items())
            or fast_p['code_sha256'][Path(steps.__file__).name] != sha(Path(steps.__file__))
            or fast_p['code_sha256'][Path(split.__file__).name] != sha(Path(split.__file__))):
        raise ValueError('sealed centered split parameters and matching replay required')
    with np.load(fast_reference/'fit.npz', allow_pickle=False) as f:
        fast = dict(mapper=torch.from_numpy(f['mapper']), decoder=(f['decoder_weight'], f['decoder_bias']),
                    scale=f['scale'])
    source = Path(p['source_prior'])
    if (any(sha(source/k) != v for k, v in p['source_assets'].items())
            or read(source/'protocol.json')['code_sha256'] != weak.code_bindings()):
        raise ValueError('frozen source shape assets changed')
    src = read(source/'result.json')
    match_scale = np.asarray(src['match_scale'])
    net = LagState(src['inputs'], src['edges']).eval().requires_grad_(False)
    with np.load(source/'model.npz', allow_pickle=False) as f:
        net.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    base, ckpt = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy().copy() for v in (base.center, base.scale))
    base.cuda().requires_grad_(False)
    frozen, shape_frozen = state_digest(base), state_digest(net)
    if frozen != locked['baseline_protocol']['source_state_sha256'] or ckpt['threshold'] != p['original']:
        raise ValueError('qualified J-TEXT parent changed')
    with np.load(reference/'readout.npz', allow_pickle=False) as f:
        coef = torch.from_numpy(f['coef']).cuda()
    thresholds, original = cal['cutoffs'], p['original']
    ids, roles, advanced = locked['ids'], io23.source_roles(), io23.advanced_times()
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    code = [Path(f) for f in (__file__, prefix.__file__, split.__file__, steps.__file__,
                             unwrap(scores).__code__.co_filename)]
    write(out/'protocol.json', dict(
        scope='EAST360_PREFIX_SUPPORT_AND_MATCHED_SPLIT_DEVELOPMENT',
        order=ids, arms=MODES, startup=STARTUP, calibration=thresholds, original=original,
        source_sha256=PARENT_SHA256, source_label_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
        source_assets=p['source_assets'], data_bindings=locked['bound'],
        reference=str(reference), reference_assets={k: sha(reference/k) for k in (
            'protocol.json', 'seal.json', 'readout.npz', 'calibration.json')},
        fast_reference=str(fast_reference), inherited_fast_fit_sha256=sha(fast_reference/'fit.npz'),
        code_sha256={f.name: sha(f) for f in code}, fit_source_ids=roles['train'], fit_cold_ids=ids[:STARTUP],
        source_val_diagnostics_only=True, fit_suffix=False, persistent_suffix_updates=0,
        width=[prefix.SHORT_VIEW, prefix.FULL_VIEW], ramp_saturates_samples=prefix.RAMP_END,
        weights='fixed scores unchanged; ramp covers prefix only; learned correction active at all points',
        fit=dict(steps=prefix.STEPS, seed=prefix.SEED, batch=prefix.BATCH,
                 correction_bound=prefix.CORRECTION, boundary_is_fixed=True),
        dynamics='reuse exact centered-slope forecast and mapper; paired on the same learned static representation'))

    train = prefix.source_records(net, roles['train'], advanced, center, scale, match_scale, 'train')
    cold = []
    for sid in ids[:STARTUP]:
        if sha(io23.path_for('EAST', sid)) != locked['hashes'][str(sid)]:
            raise ValueError('cold raw input changed')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        mapped = prepare_actual(raw, STARTUP+1, locked['mapping'])
        cold.append(prefix.cold_record(base, net, raw, mapped, center, scale, match_scale, coef))
    layer = prefix.fit(train, cold, net, coef[-25:-1], thresholds['0.2'])
    val = prefix.source_records(net, roles['val'], advanced, center, scale, match_scale, 'val')
    held = prefix.held_source(val, layer, net)
    fit = dict(train_rows=len(train['risk']), train_pos=int((train['risk']>0.).sum()),
               short_train_rows=int((train['samples']<128).sum()),
               cold_rows={str(r['sid']): len(r['samples']) for r in cold},
               val=held, correction_weight_norm=float(layer.weight.norm()))
    write(out/'fit.json', fit)
    with (out/'prefix.npz').open('xb') as f:
        np.savez(f, weight=layer.weight.detach().numpy(), bias=layer.bias.detach().numpy())
    assets = {k: sha(out/k) for k in ('fit.json', 'prefix.npz')}
    del train, cold, val
    print(dict(stage='source_held', val=held), flush=True)
    hashes, prefix_checks, dynamics = {}, {}, {}
    args = (center, scale, match_scale, coef, fast)
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for i, sid in enumerate(ids, 1):
            name = f'{i:04d}_{sid}.npz'
            archive = reference/'shots'/name
            if sha(archive) != seal['curve_sha256'][name]:
                raise ValueError(f'changed static archive: {sid}')
            with np.load(archive, allow_pickle=False) as f:
                t, fixed = f['times'], f['fixed']
            if i <= STARTUP:
                curves = dict.fromkeys(MODES, fixed)
            else:
                if sha(io23.path_for('EAST', sid)) != locked['hashes'][str(sid)]:
                    raise ValueError(f'changed raw signal: {sid}')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'], t)
                mapped = prepare_actual(raw, i, locked['mapping'])
                curves, dynamics[str(sid)] = encode(base, net, layer, mapped, *args)
                np.testing.assert_array_equal(curves['fixed'], fixed)
                np.testing.assert_array_equal(curves['ramp'][159:], fixed[159:])
                for k in MODES[3:]:
                    n = split.LATE_EFFECT if k == 'delayed' else split.EARLY_EFFECT
                    np.testing.assert_array_equal(curves[k][:n], curves['learned'][:n])
                if i in (21, 120, 360):
                    prefix_checks[str(sid)] = {}
                    for length in (116, 143, 224):
                        x = {k: mapped[k][:length] for k in ('values','available')}
                        a, _ = encode(base, net, layer, x, *args)
                        for k, v in a.items():
                            ref = curves[k][:len(v)]
                            np.testing.assert_allclose(v, ref, rtol=0., atol=2e-6)
                            for tau in thresholds.values():
                                np.testing.assert_array_equal(v>=tau, ref>=tau)
                        prefix_checks[str(sid)][str(length)] = {
                            k: float(np.max(np.abs(v-curves[k][:len(v)]))) for k,v in a.items()}
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f, times=t, **curves)
            hashes[name] = sha(path)
            if i % 60 == 0:
                print(dict(stage='sealed', shots=i, seconds=time.monotonic()-started), flush=True)
    if (state_digest(base) != frozen or state_digest(net) != shape_frozen
            or any(sha(out/k) != v for k,v in assets.items())):
        raise ValueError('frozen state changed during replay')
    write(out/'seal.json', dict(curve_sha256=hashes, fit_assets=assets, prefix_witnesses=prefix_checks))
    write(out/'dynamics_by_shot.json', dynamics)
    per = {c: {k: [] for k in MODES} for c in CAPS}
    for i,sid in enumerate(ids, 1):
        name = f'{i:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path) != hashes[name]:
            raise ValueError('post-seal curve changed')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as f:
            for c,g in per.items():
                for k in MODES:
                    rec = score_curve(f['times'], f[k], d, down, original if i <= STARTUP else thresholds[c])
                    g[k].append(dict(shot=sid, disruptive=d, down_s=down, **rec))
    old = read(reference/'per_shot.json')
    if any(g['fixed'] != old[c]['fixed'] for c,g in per.items()):
        raise ValueError('sealed baseline outcomes differ')
    metrics = {c: {k: timing_decomposition(rows) for k,rows in g.items()} for c,g in per.items()}
    write(out/'per_shot.json', per)
    write(out/'result.json', dict(
        status='COMPLETED_EAST_PREFIX_SUPPORT_AND_SPLIT_DEVELOPMENT', metrics=metrics,
        static_vs_fixed={c: {k: flips(g['fixed'],g[k]) for k in MODES[1:3]} for c,g in per.items()},
        adaptation_vs_learned={c: {k: flips(g['learned'],g[k]) for k in MODES[3:]} for c,g in per.items()},
        real_vs_controls={c: {k: flips(g[k],g['split']) for k in MODES[4:]} for c,g in per.items()},
        seconds=time.monotonic()-started))
    print({c: {k: (v['tp'], v['fp']) for k,v in g.items()} for c,g in metrics.items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference', type=Path, required=True)
    cli.add_argument('--fast-reference', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    a = cli.parse_args()
    run(a.reference.resolve(), a.fast_reference.resolve(), a.output.resolve())
