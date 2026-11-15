"""Separate paired profile motion from channel-specific roughness feedback."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_synchronized_roughness_20261110 as sync
from . import io23, simple_data
from .commissioning_event_roughness_20261110 import (
    COLUMNS, CUE, FIRST_EFFECT, cue_views, directional_features, encode, predict, roughness,
)
from .commissioning_intra_20261110 import CAPS, STARTUP
from .commissioning_radiation_reference_20261110 import RADIATION, REFERENCE
from .commissioning_transfer_20261108 import capacity
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


ABLATION = io23.ROOT / 'probe_outputs/commissioning_radiation_reference360_20261110_v1'
PAIR = RADIATION
NAMES = ('static', 'raw_profile', 'raw_matched', 'aligned', 'shifted', 'stale')


def pair_view(static, pair):
    n = len(CUE)
    if static.ndim != 2 or static.shape[1] != 2*n or pair.shape != (len(static), 2):
        raise ValueError('static and paired cue shape or time axis mismatch')
    out = static.copy()
    out[:, [*PAIR, *(n+j for j in PAIR)]] = directional_features(pair)
    return out


def run(out):
    configure()
    started = time.monotonic()
    inputs = load_inputs()
    ids = inputs['ids']
    proto, seal = (read(REFERENCE / k) for k in ('protocol.json', 'seal.json'))
    old_protocol, old_seal = (read(ABLATION / k) for k in ('protocol.json', 'seal.json'))
    bindings = seal['artifact_sha256']
    if (proto['order'] != ids or old_protocol['order'] != ids
            or proto['source_sha256'] != PARENT_SHA256
            or old_protocol['model_and_threshold_sha256'] != bindings
            or any(sha(REFERENCE / k) != v for k, v in bindings.items())):
        raise ValueError('unchanged parent, calibration and fixed ordered cohort required')
    thresholds = read(REFERENCE / 'calibration.json')['shared_cutoffs']
    original = proto['original_threshold']
    if thresholds != old_protocol['suffix_thresholds'] or original != old_protocol['original']:
        raise ValueError('threshold mismatch')
    with np.load(REFERENCE / 'parameters.npz', allow_pickle=False) as f:
        prior = f['prior'], f['supported']
        axes = {'event': f['event_axis']}
        heads = {'event': torch.from_numpy(f['event_risk_head']).cuda()}
    index = list(PAIR)
    columns = [COLUMNS[j] for j in PAIR]
    log_ref = prior[0][index]
    if not np.all(prior[1][index]):
        raise ValueError('both profile coordinates must have an original reference')
    base, checkpoint = load_parent(DEFAULT_PARENT)
    center, scale = (a.detach().numpy().copy() for a in (base.center, base.scale))
    base.cuda()
    frozen = state_digest(base)
    if checkpoint['threshold'] != original or frozen != inputs['baseline_protocol']['source_state_sha256']:
        raise ValueError('qualified source changed')

    def verified_signal(rank, sid):
        if sha(io23.path_for('EAST', sid)) != inputs['hashes'][str(sid)]:
            raise ValueError(f'EAST waveform changed: {sid}')
        return simple_data.read_shot('EAST', sid, rate_contract='nominal')

    # Only already-sealed startup shots supply labels to this reference fit.
    safe, references = [], []
    for i, sid in enumerate(ids[:STARTUP], 1):
        name = f'{i:04d}_{sid}.npz'
        prior_path = ABLATION / 'shots' / name
        if sha(prior_path) != old_seal['curve_sha256'][name]:
            raise ValueError('startup prediction not sealed')
        disrupt, _ = io23.read_outcome('EAST', sid)
        if not disrupt:
            raw = verified_signal(i, sid)
            with np.load(prior_path, allow_pickle=False) as q:
                np.testing.assert_array_equal(q['times'], raw['times'])
            signal = prepare_actual(raw, STARTUP+1, inputs['mapping'])
            safe.append(sid)
            references.append(roughness.coordinates(signal['values'][:, columns], signal['available'][:, columns],
                                                    center[columns], scale[columns]))
    if len(safe) != 12:
        raise ValueError('expected the 12 safe startup shots')
    offsets, reference_receipt = sync.fit_reference(references, log_ref)

    out.mkdir(parents=True)
    (out / 'shots').mkdir()
    write(out / 'protocol.json', dict(scope='EAST360_DEVELOPMENT_PAIRED_PROFILE_FEEDBACK',
        reference=str(REFERENCE), ablation=str(ABLATION), order=ids, startup=STARTUP, source_sha256=PARENT_SHA256,
        model_and_threshold_sha256=bindings, original=original, suffix_thresholds=thresholds, arms=NAMES,
        pair=[CUE[j] for j in PAIR], safe_reference_ids=safe, log_ref=log_ref.tolist(),
        target_baseline_log_offsets=offsets.tolist(), reference_receipt=reference_receipt,
        persistent_updates=0, producer_sha256=sha(Path(__file__)), core_sha256=sha(Path(sync.__file__))))
    print(dict(stage='frozen_pair_reference', safe_shots=len(safe), seconds=time.monotonic()-started), flush=True)
    variants = {k: ('event', k) for k in NAMES}

    def trace(signal):
        record = encode(base, signal, center, scale)
        baseline = cue_views(record, prior, ('static',))[0]['static']
        z, m = (v[:, index] for v in record[1])
        raw = roughness.summarize(z, m, (log_ref, np.ones(2, bool)), mode='real')[0]
        views, stats = sync.summarize_pair(z, m, log_ref, offsets)
        np.testing.assert_array_equal(pair_view(baseline, views['static']), baseline)
        all_views = {k: pair_view(baseline, views[k]) for k in NAMES[2:]}
        all_views.update(static=baseline, raw_profile=pair_view(baseline, raw))
        return predict(record, all_views, axes, heads, variants), stats

    seals, audits, prefix_max = {}, {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome before sealing')):
        for i, sid in enumerate(ids, 1):
            name = f'{i:04d}_{sid}.npz'
            ref = ABLATION / 'shots' / name
            if sha(ref) != old_seal['curve_sha256'][name]:
                raise ValueError('original ablation seal mismatch')
            with np.load(ref, allow_pickle=False) as q:
                times, static, raw_profile = (q[k] for k in ('times', 'static', 'only_profiles'))
            if i <= STARTUP:
                curves = dict.fromkeys(NAMES, static)
            else:
                raw = verified_signal(i, sid)
                np.testing.assert_array_equal(times, raw['times'])
                signal = prepare_actual(raw, i, inputs['mapping'])
                curves, audits[str(sid)] = trace(signal)
                np.testing.assert_array_equal(curves['static'], static)
                np.testing.assert_array_equal(curves['raw_profile'], raw_profile)
                for x in curves.values():
                    np.testing.assert_array_equal(x[:FIRST_EFFECT], static[:FIRST_EFFECT])
                if i in (21, 120, 360):
                    prefix, _ = trace({k: signal[k][:211] for k in ('values', 'available')})
                    for k, v in prefix.items():
                        np.testing.assert_allclose(v, curves[k][:len(v)], rtol=0, atol=1e-5)
                        for cutoff in thresholds.values():
                            np.testing.assert_array_equal(v >= cutoff, curves[k][:len(v)] >= cutoff)
                    prefix_max[str(sid)] = max(float(np.max(abs(v-curves[k][:len(v)]))) for k, v in prefix.items())
            path = out / 'shots' / name
            with path.open('xb') as f:
                np.savez_compressed(f, times=times, **curves)
                f.flush()
                os.fsync(f.fileno())
            seals[name] = sha(path)
            if i % 60 == 0:
                print(dict(stage='sealed', shots=i, seconds=time.monotonic()-started), flush=True)
    if state_digest(base) != frozen:
        raise AssertionError('source changed during replay')
    write(out / 'seal.json', dict(curve_sha256=seals, prefix_max=prefix_max))
    write(out / 'block_diagnostics.json', audits)

    per = {c: {k: [] for k in NAMES} for c in CAPS}
    maxima, labels = {k: [] for k in NAMES}, []
    for i, sid in enumerate(ids, 1):
        name = f'{i:04d}_{sid}.npz'
        path = out / 'shots' / name
        if sha(path) != seals[name]:
            raise ValueError('prediction changed after seal')
        d, down = io23.read_outcome('EAST', sid)
        labels.append(d)
        with np.load(path, allow_pickle=False) as q:
            times = q['times']
            eligible = np.round((down-times)*1000., 9) > 30. if d else np.ones(len(times), bool)
            for k in NAMES:
                v = q[k]
                maxima[k].append(float(v[eligible].max()) if eligible.any() else -np.inf)
                for c, g in per.items():
                    g[k].append(dict(shot=sid, disruptive=d, down_s=down,
                                     **score_curve(times, v, d, down, original if i <= STARTUP else thresholds[c])))
    ref = read(ABLATION / 'per_shot.json')
    if any(g[k] != ref[c][v] for c, g in per.items() for k, v in
           (('static', 'static'), ('raw_profile', 'only_profiles'))):
        raise AssertionError('reference alarms changed')
    metrics = {c: {k: timing_decomposition(v) for k, v in g.items()} for c, g in per.items()}
    paired = {c: {k: flips(g['static'], g[k]) for k in NAMES[1:]} for c, g in per.items()}
    attribution = {c: {k: flips(g[k], g['aligned']) for k in ('raw_matched', 'shifted', 'stale')}
                   for c, g in per.items()}
    write(out / 'per_shot.json', per)
    write(out / 'result.json', dict(status='COMPLETED_DEVELOPMENT_PAIRED_PROFILE_FEEDBACK', metrics=metrics,
        paired=paired, aligned_vs_controls=attribution, posthoc_capacity_not_deployable={
            k: capacity(v, labels, original, startup=STARTUP) for k, v in maxima.items()},
        seconds=time.monotonic()-started))
    print({c: {k: (v['tp'], v['fp']) for k, v in g.items()} for c, g in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
