"""Can past-only inputs distinguish the supplied offline event families?

This is a CONDITIONAL family-classification probe, not an event detector, an
online update, or a disruption predictor. Unannotated intervals remain unknown.
The qualified risk model is immutable. The supplied teacher used whole records
and overlaps historical source splits: all measurements are DEVELOPMENT ONLY.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from . import simple_data, io23
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent, sha256


LABEL_SHA256 = '24fa872346d5b269f2a1ff3d750a8d810c1bc66195c47c34bd665c9da00b197c'
DEFAULT_LABELS = io23.ROOT / 'outputs/event_labels_audit_20261019/input/event_labels_20261018/JTEXT_event_labels.csv'


@dataclass(frozen=True)
class Event:
    shot: int
    index: int
    family: int
    start: float
    end: float


def assign_events(times, events):
    """Offline targets on an absolute clock; -1 means UNKNOWN, never normal."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('finite increasing diagnostic clock required')
    targets = np.full(len(times), -1, dtype=np.int64)
    ids = np.full(len(times), -1, dtype=np.int64)
    previous = None
    # CSV boundaries were exported to nanosecond decimal precision. Rounding
    # only this comparison avoids 0.15+i/1000 binary arithmetic changing edges;
    # it does not resample or modify any input signal/time feature.
    compared = np.round(times, 9)
    for e in events:
        if (e.index < 1 or e.family not in range(6) or not np.isfinite([e.start, e.end]).all()
                or e.end <= e.start or (previous is not None and (
                    e.shot != previous.shot or e.index != previous.index + 1 or e.start < previous.end))):
            raise ValueError('invalid, unordered, overlapping or mixed-shot event intervals')
        selected = (compared >= e.start) & (compared < e.end)
        targets[selected], ids[selected] = e.family, e.index
        previous = e
    return targets, ids


def read_events(path):
    grouped, metadata = defaultdict(list), {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        expected = {'shot_no', 'cohort', 'disruption_label', 'event_index', 'event_class',
                    'event_start_s', 'event_end_s', 'annotation_status'}
        if set(reader.fieldnames or []) != expected:
            raise ValueError('unexpected event-label schema')
        for r in reader:
            shot, outcome, status = int(r['shot_no']), int(r['disruption_label']), r['annotation_status']
            meta = dict(cohort=r['cohort'], outcome=outcome, status=status)
            if (outcome not in (0, 1) or status not in ('event', 'no_event_detected', 'excluded_short_record')
                    or r['cohort'] not in ('training_pool', 'sanity')
                    or (shot in metadata and metadata[shot] != meta)):
                raise ValueError('inconsistent shot metadata')
            metadata[shot] = meta
            if status == 'event':
                if r['event_class'] not in tuple('ABCDEF'):
                    raise ValueError('unknown event family')
                grouped[shot].append(Event(shot, int(r['event_index']), ord(r['event_class']) - ord('A'),
                                           float(r['event_start_s']), float(r['event_end_s'])))
            elif any(r[k] for k in ('event_index', 'event_class', 'event_start_s', 'event_end_s')):
                raise ValueError('unannotated interval must not contain a target')
    for events in grouped.values():
        if events[0].index != 1:
            raise ValueError('event numbering must start at one')
        assign_events(np.array([0.]), events)
    return dict(grouped), metadata


def causal_summary(x, available, center, scale):
    """115 inputs: last/mean/std/half-window change/mask fraction, 23 each."""
    valid = available & torch.isfinite(x)
    clean = torch.where(valid, x, center)
    z = torch.asinh((clean - center) / scale)
    count = valid.sum(1).clamp_min(1)
    mean = z.sum(1) / count
    variance = ((z - mean[:, None]).square() * valid).sum(1) / count
    indices = torch.arange(x.shape[1], device=x.device)[None, :, None]
    last_index = torch.where(valid, indices, -1).amax(1)
    last = z.gather(1, last_index.clamp_min(0)[:, None]).squeeze(1)
    last = last.masked_fill(last_index < 0, 0.)
    split = x.shape[1] // 2
    first = z[:, :split].sum(1) / valid[:, :split].sum(1).clamp_min(1)
    second = z[:, split:].sum(1) / valid[:, split:].sum(1).clamp_min(1)
    return torch.cat((last, mean, variance.sqrt(), second - first, valid.float().mean(1)), -1)


def frozen_features(model, x, available):
    """Read internal tokens with a temporary hook; no labels or outcomes enter."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('immutable eval-mode parent required')
    captured = []

    def capture(module, args):
        captured.append(args[0].detach().reshape(len(x), -1, model.cfg.dim))

    hook = model.forecast_head.register_forward_pre_hook(capture)
    try:
        # Match the existing token / FFN gradient execution path, not the fused
        # no-grad Transformer path (which has a tested CUDA numerical difference).
        with torch.enable_grad():
            fast = model.initial_fast(len(x)).detach().requires_grad_(True)
            output = model(x, available, fast)
        if len(captured) != 1:
            raise AssertionError('unexpected forecast-head call count')
        tokens = captured[0]
        valid = available.reshape(len(x), model.cfg.window // model.cfg.patch, -1).any(-1)
        mean = (tokens[:, :-1] * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return torch.cat((tokens[:, -1], mean, tokens[:, -2]), -1), output['logit'].detach()[:, None]
    finally:
        hook.remove()


def balanced_event_weights(event_ids):
    event_ids = np.asarray(event_ids)
    if event_ids.ndim != 1 or not len(event_ids):
        raise ValueError('nonempty event identifiers required')
    _, inverse, counts = np.unique(event_ids, return_inverse=True, return_counts=True)
    return 1. / counts[inverse]


def shuffle_event_targets(y, event_ids, strata, *, seed):
    """Permute whole-event families within declared TRAIN-only nuisance strata.

    Preserves event counts by class within each stratum, not endpoint counts.
    Outcome/campaign strata enter ONLY this negative control, never model input.
    """
    y, event_ids, strata = map(np.asarray, (y, event_ids, strata))
    if y.ndim != 1 or y.shape != event_ids.shape or y.shape != strata.shape:
        raise ValueError('matching endpoint arrays required')
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    event_label = {}
    for eid in np.unique(event_ids):
        rows = event_ids == eid
        if len(set(y[rows])) != 1 or len(set(strata[rows])) != 1:
            raise ValueError('event has conflicting class or stratum')
        groups[int(strata[rows][0])].append(eid)
        event_label[eid] = int(y[rows][0])
    result = y.copy()
    for eids in groups.values():
        labels = rng.permutation([event_label[eid] for eid in eids])
        for eid, label in zip(eids, labels):
            result[event_ids == eid] = label
    return result


def extract_cache(model, labels, output, *, stride, batch_size, device):
    grouped, _ = read_events(labels)
    roles = simple_data.source_roles()
    rolemap = {int(s): r for r, shots in roles.items() for s in shots}
    collected = defaultdict(list)
    next_event_id, files = 0, {}
    selected_shots = sorted(set(grouped) & set(rolemap))
    for number, sid in enumerate(selected_shots, 1):
        shot = simple_data.read_shot('J-TEXT', sid)
        # CSV provides interval targets, never the hard outcome, including for
        # shuffled-control strata. This extraction is explicitly offline.
        hdf_outcome, _ = io23.read_outcome('J-TEXT', sid)
        y, event_ids = assign_events(shot['times'], grouped[sid])
        endpoints = np.arange(0, len(shot['times']), stride)
        endpoints = endpoints[y[endpoints] >= 0]
        eidmap = {e.index: next_event_id + i for i, e in enumerate(grouped[sid])}
        starts = {e.index: e.start for e in grouped[sid]}
        next_event_id += len(grouped[sid])
        files[str(sid)] = sha256(io23.path_for('J-TEXT', sid))
        for lo in range(0, len(endpoints), batch_size):
            ends = endpoints[lo:lo + batch_size]
            windows = [simple_data.window(shot, int(e), model.cfg.window) for e in ends]
            x = torch.as_tensor(np.stack([w[0] for w in windows]), device=device)
            mask = torch.as_tensor(np.stack([w[1] for w in windows]), device=device)
            latent, risk = frozen_features(model, x, mask)
            raw = causal_summary(x, mask, model.center, model.scale)
            collected['latent'].append(latent.cpu().numpy())
            collected['raw'].append(raw.cpu().numpy())
            collected['risk_only'].append(risk.cpu().numpy())
            collected['clock'].append((shot['times'][ends] - shot['times'][0])[:, None].astype('float32'))
            collected['target'].append(y[ends])
            collected['shot'].append(np.full(len(ends), sid, dtype=np.int64))
            collected['endpoint'].append(ends)
            collected['event_id'].append(np.array([eidmap[int(i)] for i in event_ids[ends]]))
            collected['event_age_ms'].append(np.array([(shot['times'][j] - starts[int(event_ids[j])]) * 1000 for j in ends]))
            collected['outcome'].append(np.full(len(ends), int(hdf_outcome), dtype=np.int64))
            collected['role'].append(np.full(len(ends), rolemap[sid], dtype='U5'))
        if number % 50 == 0 or number == len(selected_shots):
            print(json.dumps(dict(stage='extract', shots=number, total=len(selected_shots))), flush=True)
    cache = {k: np.concatenate(v) for k, v in collected.items()}
    if not all(np.isfinite(cache[k]).all() for k in ('latent', 'raw', 'risk_only', 'clock')):
        raise FloatingPointError('nonfinite causal features')
    np.savez_compressed(output / 'features.npz', **cache)
    (output / 'signal_hashes.json').write_text(json.dumps(files, indent=2))
    return cache


def evaluate(y, pred, event_ids, shots, ages):
    weight = balanced_event_weights(event_ids)
    by_shot = {str(int(s)): float(np.mean(pred[shots == s] == y[shots == s])) for s in np.unique(shots)}
    by_class = {c: dict(endpoints=int((y == c).sum()), events=int(len(np.unique(event_ids[y == c]))),
                       shots=int(len(np.unique(shots[y == c])))) for c in range(6)}
    age_bins = {}
    for lo, hi in ((0, 10), (10, 30), (30, 100), (100, float('inf'))):
        use = (ages >= lo - 1e-7) & (ages < hi - 1e-7)
        age_bins[f'{lo}_{hi}'] = dict(n=int(use.sum()), accuracy=float((pred[use] == y[use]).mean()) if use.any() else None)
    return dict(endpoints=len(y), events=len(np.unique(event_ids)), shots=len(by_shot),
        endpoint_accuracy=float(np.mean(y == pred)),
        event_weighted_accuracy=float(np.average(y == pred, weights=weight)),
        event_weighted_macro_f1=float(f1_score(y, pred, labels=list(range(6)), sample_weight=weight, average='macro', zero_division=0)),
        event_weighted_macro_recall=float(recall_score(y, pred, labels=list(range(6)), sample_weight=weight, average='macro', zero_division=0)),
        shot_equal_accuracy=float(np.mean(list(by_shot.values()))), per_shot_accuracy=by_shot,
        confusion_event_weighted=confusion_matrix(y, pred, labels=list(range(6)), sample_weight=weight).tolist(),
        support=by_class, age_bins=age_bins)


def fit_probe(cache, *, feature, seed, shuffled, steps, device):
    train = cache['role'] == 'train'
    raw_y = cache['target'][train]
    event_ids = cache['event_id'][train]
    # Strata come from TRAIN shot IDs only. No test labels or moments fit a
    # model or select its epoch. They retain outcome and coarse campaign mix in
    # the shuffled-label control without giving either to the predictor.
    boundaries = np.quantile(np.unique(cache['shot'][train]), [.25, .5, .75])
    strata = np.searchsorted(boundaries, cache['shot'][train]) + 4 * cache['outcome'][train]
    y = shuffle_event_targets(raw_y, event_ids, strata, seed=seed) if shuffled else raw_y.copy()
    weights = balanced_event_weights(event_ids)
    counts = np.bincount(y, weights=weights, minlength=6)
    if (counts <= 0).any():
        raise ValueError('all six families need source TRAIN support')
    sampling = weights / counts[y]
    sampling /= sampling.sum()
    train_x = cache[feature][train].astype(np.float64)
    mean = np.average(train_x, weights=weights, axis=0)
    scale = np.sqrt(np.average((train_x - mean) ** 2, weights=weights, axis=0))
    scale = np.maximum(scale, 1e-6)
    x = torch.as_tensor(((cache[feature] - mean) / scale).astype('float32'), device=device)
    xx, yy = x[torch.as_tensor(train, device=device)], torch.as_tensor(y, device=device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = nn.Linear(x.shape[1], 6).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.001)
    first_loss = None
    for step in range(steps):
        idx = torch.as_tensor(rng.choice(len(y), 512, replace=True, p=sampling), device=device)
        loss = F.cross_entropy(head(xx[idx]), yy[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite probe loss')
        if first_loss is None:
            first_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = head(x).argmax(-1).cpu().numpy()
    results = {}
    for role in ('val', 'test'):
        use = cache['role'] == role
        results[role] = evaluate(cache['target'][use], prediction[use], cache['event_id'][use], cache['shot'][use], cache['event_age_ms'][use])
    return dict(feature=feature, seed=seed, shuffled=shuffled, steps=steps,
                initial_train_loss=first_loss, final_train_loss=float(loss.detach()),
                shuffled_endpoint_fraction=float(np.mean(raw_y != y)), results=results), dict(
                    weight=head.weight.detach().cpu(), bias=head.bias.detach().cpu(),
                    mean=torch.from_numpy(mean), scale=torch.from_numpy(scale)), prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--seeds', type=int, nargs='+', default=[131, 132, 133])
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.output.exists() or min(args.steps, args.stride) < 1 or sha256(args.labels) != LABEL_SHA256:
        parser.error('new output, positive budgets and exact audited event CSV required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested local CUDA is unavailable')
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    base, _ = load_parent(DEFAULT_PARENT)
    original = {k: v.detach().clone() for k, v in base.state_dict().items()}
    base.to(args.device)
    protocol = dict(scope='DEVELOPMENT_ONLY_CONDITIONAL_EVENT_FAMILY_PROBE',
        source_parent_sha256=PARENT_SHA256, labels_sha256=LABEL_SHA256, window=128,
        hard_outcome_authority='HDF5 IsDisrupt; CSV supplies interval targets only',
        feature_input='past window only; no outcome/event class/end/duration as features',
        background='unknown, excluded from this conditional-family task, not normal',
        teacher='offline whole-record K6; teacher provenance overlap with historical TEST220',
        event_start_is_online_detection_time=False, east_online_gain_measured=False,
        trained_parameters='new linear probe only; no original parameters modified',
        roles='canonical source TRAIN1353 fit; VAL160 and TEST220 diagnostic reporting only',
        standardizer='source TRAIN annotated endpoints, each event has equal total weight',
        sampling='class-balanced and event-balanced TRAIN endpoints',
        shuffle='permute event class within TRAIN outcome x shot-number quartile; unknown intervals excluded',
        selection='fixed steps; no validation/test early stopping or hyperparameter selection',
        code_sha256=sha256(Path(__file__)), steps=args.steps, seeds=args.seeds,
        stride=args.stride, batch_size=512, device=args.device, torch_version=torch.__version__)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    cache = extract_cache(base, args.labels, args.output, stride=args.stride, batch_size=128, device=args.device)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value.cpu(), original[name], rtol=0, atol=0)
    runs, checkpoints = [], {}
    for feature, shuffled in [('clock', False), ('risk_only', False), ('raw', False), ('latent', False), ('latent', True)]:
        for seed in args.seeds:
            result, checkpoint, prediction = fit_probe(cache, feature=feature, seed=seed, shuffled=shuffled, steps=args.steps, device=args.device)
            name = f'{feature}_{"shuffled" if shuffled else "true"}_{seed}'
            checkpoints[name] = checkpoint
            np.save(args.output / f'{name}_predictions.npy', prediction)
            runs.append(result)
            (args.output / 'results.json').write_text(json.dumps(runs, indent=2, ensure_ascii=False))
            print(json.dumps(dict(stage='fit', name=name, test_shot_accuracy=result['results']['test']['shot_equal_accuracy'],
                                  test_macro_f1=result['results']['test']['event_weighted_macro_f1'])), flush=True)
    torch.save(checkpoints, args.output / 'probe_heads.pt')
    if sha256(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('qualified source file changed')
    summary = dict(status='COMPLETE_PROBE_ONLY', elapsed_seconds=time.monotonic() - started,
        source_file_and_parameters_unchanged=True, rows=len(cache['target']),
        events=int(len(np.unique(cache['event_id']))), shots=int(len(np.unique(cache['shot']))),
        feature_cache_sha256=sha256(args.output / 'features.npz'),
        results_sha256=sha256(args.output / 'results.json'),
        no_east_signals_or_labels_read=True, no_online_updates=True)
    (args.output / 'completed.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
