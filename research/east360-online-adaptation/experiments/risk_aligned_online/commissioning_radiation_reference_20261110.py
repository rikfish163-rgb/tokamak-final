"""Isolate the observed radiation-profile contribution of the roughness update."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import io23, simple_data
from .commissioning_event_roughness_20261110 import CUE, FIRST_EFFECT, cue_views, encode, predict
from .commissioning_intra_20261110 import CAPS, STARTUP
from .commissioning_transfer_20261108 import capacity
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


REFERENCE = io23.ROOT / 'probe_outputs/commissioning_event_roughness360_20261110_v1'
RADIATION = tuple(CUE.index(k) for k in ('sxr_var', 'xuv_var'))
NAMES = ('static', 'real', 'hold_profiles', 'only_profiles', 'hold_profile_persistence')


def mixed_features(static, real):
    n = len(CUE)
    if static.ndim != 2 or real.shape != static.shape or static.shape[1] != 2*n:
        raise ValueError('static and adapted event feature shape must match')
    lasting = [n+j for j in RADIATION]
    radiation = [*RADIATION, *lasting]
    hold, only, hold_lasting = real.copy(), static.copy(), real.copy()
    hold[:, radiation] = static[:, radiation]
    only[:, radiation] = real[:, radiation]
    hold_lasting[:, lasting] = static[:, lasting]
    return dict(static=static, real=real, hold_profiles=hold, only_profiles=only,
                hold_profile_persistence=hold_lasting)


def run(out):
    configure()
    started = time.monotonic()
    proto, ref_seal = (read(REFERENCE / k) for k in ('protocol.json', 'seal.json'))
    refs = ref_seal['artifact_sha256']
    if any(sha(REFERENCE / k) != refs[k] for k in ('parameters.npz', 'calibration.json')):
        raise ValueError('reference parameters or threshold changed')
    inputs = load_inputs()
    ids = inputs['ids']
    if ids != proto['order'] or proto['source_sha256'] != PARENT_SHA256:
        raise ValueError('reference source and fixed order required')
    original = proto['original_threshold']
    thresholds = read(REFERENCE / 'calibration.json')['shared_cutoffs']
    with np.load(REFERENCE / 'parameters.npz', allow_pickle=False) as data:
        prior = (data['prior'], data['supported'])
        axes = {'event': data['event_axis']}
        heads = {'event': torch.from_numpy(data['event_risk_head']).cuda()}
    base, checkpoint = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy().copy() for v in (base.center, base.scale))
    base.cuda()
    before = state_digest(base)
    if checkpoint['threshold'] != original or before != inputs['baseline_protocol']['source_state_sha256']:
        raise ValueError('source mismatch')
    out.mkdir(parents=True)
    (out / 'shots').mkdir()
    variants = {k: ('event', k) for k in NAMES}
    write(out / 'protocol.json', dict(scope='EAST360_DEVELOPMENT_MECHANISM_ABLATION_NO_REFIT',
        reference=str(REFERENCE), order=ids, startup=STARTUP, source_sha256=PARENT_SHA256,
        hold_channels=[CUE[j] for j in RADIATION], original=original, suffix_thresholds=thresholds,
        model_and_threshold_sha256=refs, producer_sha256=sha(Path(__file__))))

    def trace(signal):
        record = encode(base, signal, center, scale)
        features, _ = cue_views(record, prior, ('static', 'real'))
        return predict(record, mixed_features(features['static'], features['real']), axes, heads, variants)

    seals, prefix_max = {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome before seal')):
        for i, sid in enumerate(ids, 1):
            name = f'{i:04d}_{sid}.npz'
            ref = REFERENCE / 'shots' / name
            if sha(ref) != ref_seal['curve_sha256'][name]:
                raise ValueError('original prediction seal mismatch')
            with np.load(ref, allow_pickle=False) as v:
                times, static, real = (v[k] for k in ('times', 'event_static', 'event_real'))
            if i <= STARTUP:
                curves = dict.fromkeys(NAMES, static)
            else:
                if sha(io23.path_for('EAST', sid)) != inputs['hashes'][str(sid)]:
                    raise ValueError('EAST waveform changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(times, raw['times'])
                signal = prepare_actual(raw, i, inputs['mapping'])
                curves = trace(signal)
                np.testing.assert_array_equal(curves['static'], static)
                np.testing.assert_array_equal(curves['real'], real)
                for curve in curves.values():
                    np.testing.assert_array_equal(curve[:FIRST_EFFECT], static[:FIRST_EFFECT])
                if i in (21, 120, 360):
                    prefix = trace({k: signal[k][:211] for k in ('values', 'available')})
                    for k, v in prefix.items():
                        np.testing.assert_allclose(v, curves[k][:len(v)], rtol=0, atol=1e-5)
                        for cutoff in thresholds.values():
                            np.testing.assert_array_equal(v >= cutoff, curves[k][:len(v)] >= cutoff)
                    prefix_max[str(sid)] = max(float(np.max(abs(v-curves[k][:len(v)]))) for k, v in prefix.items())
            path = out / 'shots' / name
            with path.open('xb') as file:
                np.savez_compressed(file, times=times, **curves)
                file.flush()
                os.fsync(file.fileno())
            seals[name] = sha(path)
            if i % 60 == 0:
                print(dict(stage='sealed', shots=i, seconds=time.monotonic()-started), flush=True)
    if state_digest(base) != before:
        raise AssertionError('source changed during replay')
    write(out / 'seal.json', dict(curve_sha256=seals, prefix_max=prefix_max))

    per = {c: {k: [] for k in NAMES} for c in CAPS}
    maxima, labels = {k: [] for k in NAMES}, []
    for i, sid in enumerate(ids, 1):
        path = out / 'shots' / f'{i:04d}_{sid}.npz'
        if sha(path) != seals[path.name]:
            raise ValueError('curve changed after seal')
        d, down = io23.read_outcome('EAST', sid)
        labels.append(d)
        with np.load(path, allow_pickle=False) as trace_:
            t = trace_['times']
            eligible = np.round((down-t)*1000., 9) > 30. if d else np.ones(len(t), bool)
            for k in NAMES:
                v = trace_[k]
                maxima[k].append(float(v[eligible].max()) if eligible.any() else -np.inf)
                for c, g in per.items():
                    g[k].append(dict(shot=sid, disruptive=d, down_s=down,
                                      **score_curve(t, v, d, down, original if i <= STARTUP else thresholds[c])))
    ref = read(REFERENCE / 'per_shot.json')
    if any(g[k] != ref[c][f'event_{k}'] for c, g in per.items() for k in ('static', 'real')):
        raise AssertionError('reference alarms changed')
    metrics = {c: {k: timing_decomposition(v) for k, v in g.items()} for c, g in per.items()}
    changed = {c: {k: flips(g['static'], g[k]) for k in NAMES[1:]} for c, g in per.items()}
    write(out / 'per_shot.json', per)
    write(out / 'result.json', dict(status='COMPLETED_EXPLORATORY_CHANNEL_ABLATION', metrics=metrics,
        paired=changed, posthoc_capacity_not_deployable={k: capacity(v, labels, original, startup=STARTUP)
        for k, v in maxima.items()}, seconds=time.monotonic()-started))
    print({c: {k: (v['tp'], v['fp']) for k, v in g.items()} for c, g in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
