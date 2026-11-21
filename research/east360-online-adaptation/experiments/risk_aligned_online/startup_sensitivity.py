"""Frozen-model startup diagnostics; no online policy or physical causality claim.

All 240 locked development shots are included. Channel interventions at fixed
early endpoints and the original first alarm are one-window sensitivity tests,
not whole-shot TP/FP flips. History tests compare the SAME later real endpoint
with varying amounts of its already observed past; never fabricate startup past.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .operating_curve_audit import load_sealed
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash
from .verify_event_projection import recount_prediction

ROOT = Path(__file__).parent
REFERENCE = ROOT/'runs/masked_prefix240_20261021'
EARLY_ENDPOINTS = (0, 7, 15, 31, 49, 63, 99, 127, 152)
HISTORY_ENDPOINTS = (127, 255, 1023)
HISTORY_LENGTHS = (8, 16, 32, 64, 128)


def _validate(x, a):
    if (x.shape != (128, 23) or a.shape != x.shape or a.dtype != bool
            or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x[a]).all()):
        raise ValueError('finite observed W128/flat23 values and boolean mask required')


def channel_intervention(values, available, center, channel, method):
    x, a, c = map(np.asarray, (values, available, center))
    _validate(x, a)
    if (c.shape != (23,) or not np.isfinite(c).all() or type(channel) is not int
            or not 0 <= channel < 23 or method not in ('missing', 'source_center')):
        raise ValueError('finite source centers, valid channel and explicit intervention required')
    out, mask = x.copy(), a.copy()
    if method == 'missing':
        mask[:, channel] = False
    else:
        out[a[:, channel], channel] = c[channel]
    return out, mask


def history_intervention(values, available, length):
    x, a = map(np.asarray, (values, available))
    _validate(x, a)
    if type(length) is not int or not 1 <= length <= 128:
        raise ValueError('history length must be an integer in [1,128]')
    mask = a.copy()
    mask[:128-length] = False
    return x.copy(), mask


def summarize_records(records, threshold):
    groups = defaultdict(list)
    for row in records:
        for variant, score in row['variants'].items():
            key = (row['device'], row['group'], row['tag'], variant)
            groups[key].append((row['shot'], row['baseline'], score))
    out = []
    for (device, group, tag, variant), entries in sorted(groups.items()):
        old = np.asarray([e[1] for e in entries], dtype=float)
        new = np.asarray([e[2] for e in entries], dtype=float)
        out.append(dict(device=device, group=group, tag=tag, variant=variant,
            n_windows=len(entries), n_shots=len({e[0] for e in entries}),
            mean_baseline=float(old.mean()), mean_changed=float(new.mean()),
            mean_delta=float((new-old).mean()), median_delta=float(np.median(new-old)),
            crossing_down=int(((old >= threshold) & (new < threshold)).sum()),
            crossing_up=int(((old < threshold) & (new >= threshold)).sum())))
    return out


def _probabilities(model, windows):
    x = np.stack([r[0] for r in windows])
    a = np.stack([r[1] for r in windows])
    result = []
    with torch.no_grad():
        for begin in range(0, len(x), 128):
            result.extend(model(torch.as_tensor(x[begin:begin+128], device='cuda'),
                torch.as_tensor(a[begin:begin+128], device='cuda'))['logit'].sigmoid().cpu().tolist())
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite diagnostic prediction')
    return result


def audit_shot(model, shot, group, threshold, *, reference=None):
    tasks = [('early_'+str(q), q, 'channel') for q in EARLY_ENDPOINTS if q < len(shot['times'])]
    tasks += [('history_'+str(q), q, 'history') for q in HISTORY_ENDPOINTS if q < len(shot['times'])]
    if reference is not None:
        hits = np.flatnonzero(shot['valid'] & (reference.astype(float) >= threshold))
        if len(hits) and hits[0] <= max(EARLY_ENDPOINTS):
            tasks.append(('first_alarm', int(hits[0]), 'channel'))
    windows, descriptors = [], []
    center = model.center.detach().cpu().numpy()
    for tag, q, kind in tasks:
        x, a = simple_data.window(shot, q, 128)
        start = len(windows)
        windows.append((x, a))
        names = []
        if kind == 'channel':
            for method in ('missing', 'source_center'):
                for channel, name in enumerate(io23.RAW23):
                    windows.append(channel_intervention(x, a, center, channel, method))
                    names.append(f'{method}:{name}')
        else:
            for length in HISTORY_LENGTHS:
                windows.append(history_intervention(x, a, length))
                names.append(f'history:{length}')
        descriptors.append((tag, q, kind, start, names))
    probabilities = _probabilities(model, windows)
    rows, max_error = [], 0.
    for tag, q, kind, start, names in descriptors:
        p = probabilities[start]
        if reference is not None:
            error = abs(p-float(reference[q]))
            max_error = max(max_error, error)
            if error > 2e-6:
                raise AssertionError(f'original sealed baseline changed: {shot["shot"]}, {q}, {error}')
        rows.append(dict(device=shot['device'], shot=shot['shot'], group=group if shot['valid'][q]
            else 'after_outcome', tag=tag, kind=kind, index=q, time_s=float(shot['times'][q]),
            record_start_s=float(shot['times'][0]), valid=bool(shot['valid'][q]),
            available_points=int(shot['available'][max(0, q-127):q+1].any(1).sum()),
            baseline=p, variants=dict(zip(names, probabilities[start+1:start+1+len(names)]))))
    return rows, max_error, len(windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output directory required')
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(131)
    started = time.monotonic()
    base, parent = load_parent(DEFAULT_PARENT)
    model = PersistentFfnModel(base).cuda().eval().requires_grad_(False)
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    threshold = float(parent['threshold'])
    order = json.loads((REFERENCE/'order.json').read_text())
    if len(order) != 240 or len({int(r['shot']) for r in order}) != 240:
        raise AssertionError('canonical distinct EAST240 order required')
    signals = [simple_data.attach_outcome(apply_input_contract(simple_data.read_shot(
        'EAST', r['shot'], rate_contract='nominal')), 'EAST', {}) for r in order]
    sealed, hashes = load_sealed(REFERENCE/'static/stream', order, signals, threshold)
    roles = simple_data.source_roles()
    train = sorted(map(int, roles['train']))
    if len(train) != 1353 or len(set(train)) != 1353 or set(train) & (set(roles['val']) | set(roles['test'])):
        raise AssertionError('source TRAIN identity and holdout separation required')
    # Source control selection is fixed before inspecting any model outputs.
    source_ids = sorted(map(int, np.random.default_rng(131).choice(train, 120, replace=False)))
    _durable_json(args.output/'protocol.json', dict(
        scope='POSTHOC DEVELOPMENT FROZEN MODEL STARTUP SENSITIVITY; NOT AN ONLINE METHOD',
        source_sha256=PARENT_SHA256, order_sha256=_hash(REFERENCE/'order.json'), threshold=threshold,
        code_sha256=_hash(Path(__file__)), early_endpoints=list(EARLY_ENDPOINTS),
        history_endpoints=list(HISTORY_ENDPOINTS), history_lengths=list(HISTORY_LENGTHS),
        source_control_ids=source_ids, source_control_selection='120 TRAIN IDs without replacement, seed131',
        source_control_is_not_independent_test=True, one_window_crossings_are_not_shot_rescues=True,
        real_past_only=True, no_optimization=True, no_physical_causality_claim=True,
        no_missing_before_record_history_fabricated=True,
        endpoint_selection='fixed endpoints plus original first alarm <=152; outcomes used only to describe groups'))
    records, reference_decisions, calls, max_error = [], [], 0, 0.
    for index, (shot, trace) in enumerate(zip(signals, sealed), 1):
        decision = recount_prediction(trace, threshold)
        reference_decisions.append(decision)
        if decision['is_disrupt']:
            group = 'disruptive_timely' if decision['timely'] else 'disruptive_missed'
        elif not decision['alarm']:
            group = 'safe_tn'
        else:
            first = np.flatnonzero(trace['valid'] & (trace['scores'].astype(float) >= threshold))[0]
            group = 'safe_early_fp' if first <= 152 else 'safe_late_fp'
        rows, error, n = audit_shot(model, shot, group, threshold, reference=trace['scores'])
        records.extend(rows); calls += n; max_error = max(max_error, error)
        if index % 40 == 0:
            print(json.dumps(dict(east_shots=index, forwards=calls, max_baseline_error=max_error)), flush=True)
    advanced = simple_data.advanced_times()
    source_composition = {'disruptive': 0, 'safe': 0}
    for index, sid in enumerate(source_ids, 1):
        shot = simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('J-TEXT', sid)),
                                         'J-TEXT', advanced)
        group = 'disruptive' if shot['is_disrupt'] else 'safe'
        source_composition[group] += 1
        rows, _, n = audit_shot(model, shot, group, threshold)
        records.extend(rows); calls += n
        if index % 40 == 0:
            print(json.dumps(dict(source_control_shots=index, forwards=calls)), flush=True)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v.detach().cpu(), frozen[k], rtol=0, atol=0)
    if _hash(DEFAULT_PARENT) != PARENT_SHA256 or model.memory.any():
        raise AssertionError('immutable source or temporary/persistent state changed')
    _durable_json(args.output/'records.json', records)
    summary = summarize_records(records, threshold)
    _durable_json(args.output/'summary.json', summary)
    _durable_json(args.output/'completed.json', dict(
        east_shots=len(order), source_control_shots=len(source_ids), source_control_composition=source_composition,
        windows_evaluated=calls, records=len(records), threshold=threshold,
        max_baseline_error=max_error, checked_seals=hashes,
        original_decisions=reference_decisions, source_and_model_bitwise_unchanged=True,
        elapsed_seconds=time.monotonic()-started, no_whole_shot_intervention_metric_claim=True,
        summary_sha256=_hash(args.output/'summary.json'), records_sha256=_hash(args.output/'records.json')))
    print(json.dumps(dict(completed=True, windows=calls, max_baseline_error=max_error,
                          elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__': main()
