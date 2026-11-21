"""Fixed-head EAST360 experiment for locally matched in-shot decoder adaptation."""

import argparse
from inspect import unwrap
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_operating_memory_20261111 as memory
from . import commissioning_signed_calibration_20261111 as calibration
from . import commissioning_split_learning_20261111 as learn
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


MODES = ('fixed', *memory.MODES)
_TEACHING_VIEWS = dict(split='matched', delayed='recent', reversed='wrong_context')
_PHYSICAL = [io23.RAW23.index(k) for k in ('ip', 'Bt')]


def estimate(track, signal, decoder, noise, physical_scale):
    q = track['q']
    context = track['design'][:, learn.WIDTH:learn.WIDTH+3] / physical_scale
    available = signal['available'][:, _PHYSICAL]
    count = np.vstack((np.zeros((1, 2), np.int64), np.cumsum(available, axis=0)))
    last = q + 1
    recent, prior = count[last]-count[last-16], count[last-16, 0]-count[last-32, 0]
    usable = (recent >= 8).all(1) & (prior >= 8)
    forecast = learn.base_forecast(track, decoder)
    return memory.replay(forecast, learn.normalized_target(track, noise), track['observed'],
                         q, context, usable, track['length'])


@torch.no_grad()
def labelled(net, raw, signal, advanced, params, *, base=None, coef=None):
    center, scale, *_ = params
    device, t = raw['device'], raw['times']
    truth = simple_data.attach_outcome(raw, device, advanced)
    disruptive = truth['is_disrupt']
    # Pairs are a teaching task in the timely interval, not replacement hard labels.
    ix = learn.pairs.endpoints(t, truth['valid'], disruptive, truth['down'], 0.,
                              first_effect=learn.LATE_EFFECT)
    ends = np.arange(len(t)) if base is not None else np.unique(ix.ravel())
    tr = learn.track(net, signal, center, scale, ends)
    states, audit = estimate(tr, signal, *params[2:])
    pos = np.searchsorted(ends, ix)
    keep = tr['matchable'][pos].all(1)
    ix, pos = ix[keep], pos[keep]
    near, row = ix[:, 1], pos[:, 1]
    ctx = tr['context'][pos]
    tail = (not disruptive) & (t[near] >= t[-1] - weak.TAIL_S) & (ctx[:, 1, 2] < 0)
    teaching = dict(pre=tr['pre'][row], key=tr['key'][row], context=ctx, tail=tail,
                    shot=np.full(len(row), raw['shot']), disruptive=np.full(len(row), disruptive),
                    official_positive=truth['y'][near].astype(bool),
                    **{k: states[v][near] for k, v in _TEACHING_VIEWS.items()})
    alarm = None
    if base is not None:
        x, _ = features(base, signal, 'cuda')
        offset = torch.from_numpy(x).cuda() @ coef[:-25] + coef[-1]
        if disruptive:
            lead = np.round((truth['down']-t)*1000., 9)
            q = np.flatnonzero(truth['valid'] & (truth['y'] > 0.) & (lead > 30.))
        else:
            fixed = offset + torch.from_numpy(tr['pre']).cuda().tanh() @ coef[-25:-1]
            peak = fixed.topk(min(len(ends), 128)).indices.cpu().numpy()
            q = np.union1d(np.arange(0, len(ends), 16), peak)
            q = np.union1d(q, np.arange(min(160, len(ends))))
        if not len(q):
            raise ValueError('no cold warning-window support')
        alarm = dict(sid=raw['shot'], disruptive=disruptive, raw_z=offset[q].cpu().numpy(),
                     pre=tr['pre'][q], key=tr['key'][q], split=states['matched'][q])
    return teaching, alarm, audit


def pair_group(rows, match_scale):
    packed = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    matched = learn.pairs.match(packed['context'], packed['disruptive'], packed['tail'], match_scale)
    return packed, matched


def source_group(net, ids, advanced, params, match_scale, stage):
    rows, audits = [], []
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        if len(raw['times']) >= learn.LATE_EFFECT:
            r, _, a = labelled(net, raw, raw, advanced, params)
            rows.append(r)
            audits.append(a)
        if i % 450 == 0 or i == len(ids):
            print(dict(stage=stage, shots=i, total=len(ids)), flush=True)
    return pair_group(rows, match_scale), aggregate(audits)


def aggregate(rows):
    count = sum(a['observed_values'] for a in rows)
    return dict(observed_values=count, released=sum(a['released'] for a in rows),
                differs_from_recent=sum(a['matched_differs_from_recent'] for a in rows),
                forecast_mse={k: sum(a['forecast_sq'][k] for a in rows)/count for k in memory.MODES})


@torch.no_grad()
def encode(base, net, signal, params, mapper, coef):
    raw, _ = features(base, signal, 'cuda')
    start = time.perf_counter()
    tr = learn.track(net, signal, *params[:2], np.arange(len(raw)))
    states, audit = estimate(tr, signal, *params[2:])
    pre, key = torch.from_numpy(tr['pre']), torch.from_numpy(tr['key'])
    views = {'fixed': pre.tanh().numpy(), **{
        k: learn.feature_view(pre, key, torch.from_numpy(v), mapper).numpy()
        for k, v in states.items()}}
    return scores(raw, views, coef, MODES), dict(audit, feature_forecast_state_ms=(
        time.perf_counter()-start)*1000.)


def run(reference, forecast_reference, out):
    configure()
    start = time.monotonic()
    locked = load_inputs()
    p, s, cal = (read(reference/k) for k in ('protocol.json', 'seal.json', 'calibration.json'))
    ids, roles, advanced = locked['ids'], io23.source_roles(), io23.advanced_times()
    if (p['scope'] != 'EAST360_PAIRED_CONTEXT_MATCHED_WEAK_SHAPE_DEVELOPMENT'
            or p['order'] != ids or p['startup'] != STARTUP or p['source_sha256'] != PARENT_SHA256
            or len(s['curve_sha256']) != len(ids)
            or any(sha(reference/k) != v for k, v in s['artifacts'].items())):
        raise ValueError('sealed fixed EAST360 reference required')
    fp, fs = (read(forecast_reference/k) for k in ('protocol.json', 'seal.json'))
    if (fp['scope'] != 'EAST360_DIRECT25_SPLIT_RELEASE_FIXED_HEAD_DEVELOPMENT'
            or Path(fp['reference']) != reference or fp['fit_source_ids'] != roles['train']
            or fp['code_sha256'][Path(learn.__file__).name] != sha(Path(learn.__file__))
            or sha(forecast_reference/'fit.npz') != fs['fit_assets']['fit.npz']):
        raise ValueError('source-only direct decoder provenance changed')
    with np.load(forecast_reference/'fit.npz', allow_pickle=False) as f:
        decoder, noise = (f['decoder_weight'], f['decoder_bias']), f['scale']
    source = Path(p['source_prior'])
    if (any(sha(source/k) != v for k, v in p['source_assets'].items())
            or read(source/'protocol.json')['code_sha256'] != weak.code_bindings()):
        raise ValueError('frozen shape producer changed')
    info = read(source/'result.json')
    net = LagState(info['inputs'], info['edges']).eval().requires_grad_(False)
    with np.load(source/'model.npz', allow_pickle=False) as f:
        net.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    base, ckpt = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy().copy() for v in (base.center, base.scale))
    base.cuda().requires_grad_(False)
    frozen, frozen_shape = state_digest(base), state_digest(net)
    original, thresholds = p['original'], cal['cutoffs']
    if frozen != locked['baseline_protocol']['source_state_sha256'] or ckpt['threshold'] != original:
        raise ValueError('qualified source checkpoint changed')
    with np.load(reference/'readout.npz', allow_pickle=False) as f:
        coef = torch.from_numpy(f['coef']).cuda()
    beta = coef[-learn.WIDTH-1:-1]
    match_scale = np.asarray(info['match_scale'])
    params = (center, scale, decoder, noise, np.maximum(.05, np.tanh(match_scale[:3])))
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    code = [Path(x) for x in (__file__, memory.__file__, learn.__file__, calibration.__file__,
                             learn.pairs.__file__, unwrap(scores).__code__.co_filename)]
    write(out/'protocol.json', dict(
        scope='EAST360_FIXED_HEAD_MATCHED_IN_SHOT_OPERATING_CONTEXT_DEVELOPMENT',
        order=ids, startup=STARTUP, arms=MODES, source_sha256=PARENT_SHA256,
        reference=str(reference), reference_assets={k: sha(reference/k) for k in (
            'protocol.json', 'seal.json', 'readout.npz', 'calibration.json')},
        source_assets=p['source_assets'], data_bindings=locked['bound'],
        source_label_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH), original=original, calibration=thresholds,
        code_sha256={f.name: sha(f) for f in code}, forecast_reference=str(forecast_reference),
        forecast_source_fit_sha256=sha(forecast_reference/'fit.npz'),
        fit_source_ids=roles['train'], fit_cold_ids=ids[:STARTUP], no_suffix_fitting=True,
        persistent_suffix_updates=0, window=memory.WINDOW, history=memory.HISTORY,
        neighbors=memory.NEIGHBORS, strength=float(memory.STRENGTH),
        context='masked causal recent Ip, Bt and signed Ip change in frozen source coordinates',
        teaching='same mapper for all arms; separation against equal-count recent/wrong contexts',
        first_effect=learn.LATE_EFFECT, earliest_context_routing_effect=learn.LATE_EFFECT+3*learn.STRIDE))
    train, training_forecast = source_group(net, roles['train'], advanced, params, match_scale, 'source_train')
    rows, alarms, cold_audit = [], [], []
    for sid in ids[:STARTUP]:
        if sha(io23.path_for('EAST', sid)) != locked['hashes'][str(sid)]:
            raise ValueError('cold evidence changed')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        signal = prepare_actual(raw, STARTUP+1, locked['mapping'])
        r, alarm, a = labelled(net, raw, signal, None, params, base=base, coef=coef)
        rows.append(r)
        alarms.append(alarm)
        cold_audit.append(a)
    cold = pair_group(rows, match_scale)
    cold_forecast = aggregate(cold_audit)
    print(dict(stage='pairs', train=len(train[1][0]), cold=len(cold[1][0])), flush=True)
    mapper = calibration.teach((*train, training_forecast), (*cold, cold_forecast),
                               alarms, beta, thresholds['0.2'])
    held, source_val = source_group(net, roles['val'], advanced, params, match_scale, 'source_val')
    write(out/'fit.json', dict(
        teaching_source_pairs=learn.pair_receipt(*train), teaching_cold_pairs=learn.pair_receipt(*cold),
        source_official_positive_anchor_pairs=int(train[0]['official_positive'][train[1][0]].sum()),
        source_val_pairs=learn.pair_receipt(*held),
        teaching_views=_TEACHING_VIEWS, held_order=learn.check_pairs(*held, mapper, beta.cpu()),
        forecast=dict(train=training_forecast, source_val=source_val, cold=cold_forecast)))
    with (out/'mapper.npz').open('xb') as f:
        np.savez(f, mapper=mapper.numpy(), context_scale=params[-1])
    fitted = {k: sha(out/k) for k in ('fit.json', 'mapper.npz')}
    hashes, witness, audit = {}, {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for i, sid in enumerate(ids, 1):
            name = f'{i:04d}_{sid}.npz'
            old_path = reference/'shots'/name
            if sha(old_path) != s['curve_sha256'][name]:
                raise ValueError('sealed static curve changed')
            with np.load(old_path, allow_pickle=False) as f:
                t, fixed = f['times'], f['fixed']
            if i <= STARTUP:
                curves = dict.fromkeys(MODES, fixed)
            else:
                if sha(io23.path_for('EAST', sid)) != locked['hashes'][str(sid)]:
                    raise ValueError('raw EAST signal changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(t, raw['times'])
                signal = prepare_actual(raw, i, locked['mapping'])
                curves, audit[str(sid)] = encode(base, net, signal, params, mapper, coef)
                np.testing.assert_array_equal(curves['fixed'], fixed)
                for k in MODES[1:]:
                    np.testing.assert_array_equal(curves[k][:learn.LATE_EFFECT], fixed[:learn.LATE_EFFECT])
                earliest = learn.LATE_EFFECT+3*learn.STRIDE
                for k in ('recent', 'wrong_context'):
                    np.testing.assert_array_equal(curves[k][:earliest], curves['matched'][:earliest])
                if i in (21, 120, 360):
                    witness[str(sid)] = {}
                    for length in (learn.LATE_EFFECT+4, earliest+11):
                        small = {k: signal[k][:length] for k in ('values', 'available')}
                        preview, _ = encode(base, net, small, params, mapper, coef)
                        for k, z in preview.items():
                            full = curves[k][:len(z)]
                            np.testing.assert_allclose(z, full, atol=2e-6, rtol=0)
                            for tau in thresholds.values():
                                np.testing.assert_array_equal(z >= tau, full >= tau)
                        witness[str(sid)][str(length)] = {
                            k: float(np.max(np.abs(v-curves[k][:len(v)]))) for k, v in preview.items()}
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f, times=t, **curves)
            hashes[name] = sha(path)
            if i % 60 == 0:
                print(dict(stage='sealed', shots=i, seconds=time.monotonic()-start), flush=True)
    if (state_digest(base) != frozen or state_digest(net) != frozen_shape
            or any(sha(out/k) != v for k, v in fitted.items())):
        raise ValueError('parent or teaching weights changed in replay')
    write(out/'seal.json', dict(curve_sha256=hashes, fit_assets=fitted, prefix_witnesses=witness))
    write(out/'forecast_audit.json', audit)
    per = {c: {k: [] for k in MODES} for c in CAPS}
    for i, sid in enumerate(ids, 1):
        name = f'{i:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path) != hashes[name]:
            raise ValueError('sealed curve changed')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as f:
            for c, group in per.items():
                for k in MODES:
                    a = score_curve(f['times'], f[k], d, down, original if i <= STARTUP else thresholds[c])
                    group[k].append(dict(shot=sid, disruptive=d, down_s=down, **a))
    old = read(reference/'per_shot.json')
    if any(g['fixed'] != old[c]['fixed'] for c, g in per.items()):
        raise ValueError('sealed comparator per-shot results changed')
    metrics = {c: {k: timing_decomposition(v) for k, v in g.items()} for c, g in per.items()}
    write(out/'per_shot.json', per)
    write(out/'result.json', dict(
        status='COMPLETED_EAST360_OPERATING_MEMORY_DEVELOPMENT', metrics=metrics,
        paired_vs_fixed={c: {k: flips(g['fixed'], g[k]) for k in MODES[1:]} for c, g in per.items()},
        matched_vs_control={c: {k: flips(g[k], g['matched']) for k in memory.MODES if k != 'matched'}
                            for c, g in per.items()},
        suffix_forecast=aggregate(list(audit.values())), seconds=time.monotonic()-start))
    print({c: {k: (v['tp'], v['fp']) for k, v in g.items()} for c, g in metrics.items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference', type=Path, required=True)
    cli.add_argument('--forecast-reference', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    a = cli.parse_args()
    run(a.reference.resolve(), a.forecast_reference.resolve(), a.output.resolve())
