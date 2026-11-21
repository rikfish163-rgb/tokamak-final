"""Full-roster read-only level/change audit, with time/mask/Bt-matched controls."""
import argparse
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..delayed_recurrent_memory_20261025.model import patch_signal
from ..east_context_operator_20261031.run import REFERENCE, REFERENCE_SHA, read
from ..lifecycle_stream_20261025.audit import classify
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import _weighted_quantile, transform
from .observations import at_endpoint, patch_diagnostics

PLAN = io23.ROOT/'LEVEL_CHANGE_AUDIT_PLAN_20261031.md'
OPERATOR = io23.ROOT/'probe_outputs/east_operator360_v1'


def describe(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return dict(n=0, q25=None, median=None, q75=None)
    q = np.quantile(values, [.25, .5, .75])
    return dict(n=len(values), q25=float(q[0]), median=float(q[1]), q75=float(q[2]))


def source_bounds(base):
    values = [[] for _ in range(23)]
    weights = [[] for _ in range(23)]
    receipts = []
    for i, sid in enumerate(simple_data.source_roles()['train']):
        s = apply_input_contract(simple_data.read_shot('J-TEXT', sid))
        flag, down = io23.read_outcome('J-TEXT', sid)
        _, valid = io23.labels_at(s['times'], flag, down, 25.)  # only preterminal validity is used
        means, complete = patch_signal({k:s[k] for k in ('values', 'available')}, base.center, base.scale)
        q, available = means.numpy(), complete.numpy().copy()
        preterminal = valid[:len(q)*8].reshape(-1, 8).all(1)
        available &= preterminal[:, None]
        for c in range(23):
            observed = q[available[:, c], c].astype(np.float64)
            if len(observed):
                values[c].append(observed)
                weights[c].append(np.full(len(observed), 1./len(observed)))
        receipts.append(dict(shot=sid, raw_sha256=sha(io23.path_for('J-TEXT', sid)),
                             patches=len(q), preterminal_patches=int(preterminal.sum()),
                             observed_per_channel=available.sum(0).tolist()))
        if (i+1) % 500 == 0:
            print(dict(stage='source_reference', shots=i+1), flush=True)
    bounds = []
    for c, name in enumerate(io23.RAW23):
        if not values[c]:
            raise ValueError('no source reference observations: '+name)
        v, w = np.concatenate(values[c]), np.concatenate(weights[c])
        lower, median, upper = _weighted_quantile(v, w, [.01, .5, .99])
        bounds.append(dict(channel=name, lower=float(lower), median=float(median), upper=float(upper),
                           observed_shots=len(values[c]), observed_patches=len(v)))
    return dict(scope='TRAIN1353_PRETERMINAL_COMPLETE_PATCH_REFERENCE_NOT_SAFE_RANGE',
                weighting='equal total weight per observed shot per channel',
                quantiles=[.01, .5, .99], checkpoint_sha256=PARENT_SHA256,
                bounds=bounds, inputs=receipts)


def snapshot(record, end):
    if not 0 <= end < record['samples']:
        raise ValueError('snapshot outside actual observed record')
    out = at_endpoint(record['diag'], int(end))
    j = out['patch_index']
    raw_bt = abs(float(record['bt'][end]))
    band = 'small_storage_value' if raw_bt < .1 else 'middle_storage_value' if raw_bt < 1 else 'regular_storage_value'
    mask = record['complete'][j] if j is not None else np.zeros(23, bool)
    out.update(endpoint=int(end), bt_band=band, available_mask=''.join(str(int(x)) for x in mask),
               learned_context_norm=(record['norm'][j] if record['norm'] is not None and j is not None else None))
    if record['norm'] is not None and out['prefix_exactly_constant']:
        assert out['learned_context_norm'] == 0., 'observed zero-change prefix had nonzero model state'
    return out


def main(root):
    if root.exists():
        raise FileExistsError(root)
    start = time.monotonic()
    torch.set_num_threads(2)
    assert sha(REFERENCE/'completed.json') == REFERENCE_SHA
    base, _ = load_parent(DEFAULT_PARENT)
    assert base.center.device.type == 'cpu'
    root.mkdir(parents=True)
    files = set(Path(__file__).parent.glob('*.py')) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            files.add(Path(module.__file__))
    protocol = dict(scope='READ_ONLY_POSTHOC_ALL360_LEVEL_AND_ARRIVED_CHANGE',
        source_checkpoint_sha256=PARENT_SHA256, parent_completed_sha256=REFERENCE_SHA,
        parent=str(REFERENCE), plan_sha256=sha(PLAN), outcomes_used_only_for_source_validity_and_analysis=True,
        no_model_training=True, no_gpu=True, no_alarm_changes=True, independent_confirmation=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(root/'protocol.json', protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f.relative_to(io23.ROOT)))
    source = source_bounds(base)
    _durable_json(root/'source_bounds.json', source)
    lower = np.array([r['lower'] for r in source['bounds']])
    upper = np.array([r['upper'] for r in source['bounds']])
    order = [r['shot'] for r in read(REFERENCE/'order.json')]
    mapping = read(REFERENCE/'input_map.json')
    assert mapping['calibration_ids'] == order[:20]
    parent_protocol = read(REFERENCE/'protocol.json')
    hashes = read(REFERENCE/'input_hashes.json')
    saved = read(REFERENCE/'per_shot.json')['cold']
    records, snapshots = [], []
    for i, sid in enumerate(order):
        assert sha(io23.path_for('EAST', sid)) == hashes[str(sid)]
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = {k:raw[k] for k in ('values', 'available')}
        if i >= 20:
            signal = transform(signal, mapping)
        means, complete = patch_signal(signal, base.center, base.scale)
        q, available = means.numpy(), complete.numpy()
        diag = patch_diagnostics(q, available, lower, upper)
        path = REFERENCE/'shots'/f'{i+1:04d}_{sid}.npz'
        assert sha(path) == read(REFERENCE/'shots'/f'{i+1:04d}_{sid}_seal.json')['sha256']
        with np.load(path, allow_pickle=False) as a:
            assert np.array_equal(a['times'], raw['times'])
            scores = a['cold'].copy()
        flag, down = io23.read_outcome('EAST', sid)
        prediction = classify(raw['times'], scores, flag, down, parent_protocol['threshold'])
        assert dict(shot=sid, is_disrupt=flag, **prediction) == saved[i]
        category = ('TP' if prediction['timely'] else 'FN') if flag else ('FP' if prediction['alarm'] else 'TN')
        norm = None
        if i >= 20:
            info = read(OPERATOR/'shots'/f'{i+1:04d}_{sid}_audit.json')
            norm = info['fast']['learned_rule']['norm']
            assert len(norm) == len(q) and np.isfinite(norm).all()
        record = dict(shot=sid, rank=i+1, category=category, samples=len(raw['times']),
                      bt=raw['values'][:, 0], complete=available, diag=diag, norm=norm, scores=scores)
        records.append(record)
        snap = snapshot(record, 128)
        mask = available[snap['patch_index']]
        outside = ((q[snap['patch_index']] < lower) | (q[snap['patch_index']] > upper)) & mask
        snapshots.append(dict(shot=sid, rank=i+1, category=category,
                              stage='source_only20' if i < 20 else 'cold_suffix340',
                              channel_available=mask.tolist(), channel_outside=outside.tolist(), **snap))
        if (i+1) % 60 == 0:
            print(dict(stage='target_audit', shots=i+1, seconds=time.monotonic()-start), flush=True)
    _durable_json(root/'snapshots128.json', snapshots)
    groups = {}
    for stage in ('source_only20', 'cold_suffix340'):
        groups[stage] = {}
        for category in ('TP', 'FN', 'FP', 'TN'):
            rows = [r for r in snapshots if r['stage'] == stage and r['category'] == category]
            groups[stage][category] = dict(shots=len(rows),
                outside_fraction=describe(r['outside_fraction'] for r in rows),
                increment_rms=describe(r['increment_rms'] for r in rows),
                learned_context_norm=describe(r['learned_context_norm'] for r in rows),
                exactly_constant_prefixes=sum(r['prefix_exactly_constant'] for r in rows))
    channel_comparison = []
    for c, name in enumerate(io23.RAW23):
        row = dict(channel=name)
        for category in ('FP', 'TN'):
            cohort = [r for r in snapshots if r['stage'] == 'cold_suffix340' and r['category'] == category]
            observed = sum(r['channel_available'][c] for r in cohort)
            outside = sum(r['channel_outside'][c] for r in cohort)
            row[category] = dict(observed=observed, outside=outside,
                                 fraction=outside/observed if observed else None)
        channel_comparison.append(row)
    controls = [r for r in records[20:] if r['category'] == 'TN']
    matched = []
    for r in records[20:]:
        if r['category'] != 'FP':
            continue
        end = int(np.flatnonzero(r['scores'] >= parent_protocol['threshold'])[0])
        case = snapshot(r, end)
        eligible = []
        for candidate in controls:
            if candidate['samples'] <= end:
                continue
            at = snapshot(candidate, end)
            if at['bt_band'] == case['bt_band'] and at['available_mask'] == case['available_mask']:
                eligible.append(dict(shot=candidate['shot'], **at))
        outside = describe(v['outside_fraction'] for v in eligible)
        change = describe(v['increment_rms'] for v in eligible)
        higher = (case['outside_fraction'] > outside['median']
                  if case['outside_fraction'] is not None and outside['median'] is not None else None)
        lower_change = (case['increment_rms'] < change['median']
                        if case['increment_rms'] is not None and change['median'] is not None else None)
        matched.append(dict(shot=r['shot'], rank=r['rank'], snapshot=case, controls=eligible,
            matched_shots=len(eligible), control_outside=outside, control_change=change,
            higher_outside=higher, lower_change=lower_change,
            both=(higher and lower_change) if higher is not None and lower_change is not None else None))
    assert len(matched) == 42 and len(controls) == 136
    _durable_json(root/'matched_first_alarm.json', matched)
    _durable_json(root/'channel_comparison128.json', channel_comparison)
    comparison = dict(false_alarm_cases=42, available_TN_shots=136,
        cases_with_controls=sum(r['matched_shots'] > 0 for r in matched),
        level_comparable=sum(r['higher_outside'] is not None for r in matched),
        change_comparable=sum(r['lower_change'] is not None for r in matched),
        higher_outside=sum(r['higher_outside'] is True for r in matched),
        lower_change=sum(r['lower_change'] is True for r in matched),
        joint_comparable=sum(r['both'] is not None for r in matched),
        joint_higher_level_lower_change=sum(r['both'] is True for r in matched),
        exact_constant_before_alarm=sum(r['snapshot']['prefix_exactly_constant'] for r in matched),
        no_arrived_feedback_before_alarm=sum(r['snapshot']['feedback_pairs'] == 0 for r in matched))
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    done = dict(status='COMPLETED_READ_ONLY_DIAGNOSTIC', no_model_changes=True,
                source_reference_shots=1353, full_stream_shots=360, groups=groups,
                matched_comparison=comparison, seconds=time.monotonic()-start,
                source_bounds_sha256=sha(root/'source_bounds.json'),
                snapshots_sha256=sha(root/'snapshots128.json'),
                matched_sha256=sha(root/'matched_first_alarm.json'),
                causal_explanation_proven=False, independent_confirmation=False)
    _durable_json(root/'completed.json', done)
    print(done, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
