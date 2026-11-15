from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional as F

from ..event_prefix_probe import assign_events
from ..delayed_recurrent_memory_20261025.model import endpoint_corrections
from ..direction_teaching_20261026.learning import objective

KINDS = ('none', 'true', 'shuffled')


def shuffle_intervals(events, train_flags, *, seed):
    """TRAIN-only interval permutation, preserving family x outcome counts."""
    rng = np.random.default_rng(seed)
    out = {sid: list(events.get(sid, [])) for sid in sorted(train_flags)}
    for flag in (False, True):
        positions = [(sid, j) for sid in out if train_flags[sid] == flag
                     for j in range(len(out[sid]))]
        families = rng.permutation([out[s][j].family for s, j in positions])
        for (sid, j), family in zip(positions, families, strict=True):
            out[sid][j] = replace(out[sid][j], family=int(family))
    return out


def family_counts(times_by_shot, events):
    counts = np.zeros(6, dtype=np.int64)
    for sid, times in times_by_shot.items():
        y, ids = assign_events(times, events.get(sid, []))
        for index in np.unique(ids[ids >= 0]):
            counts[int(y[ids == index][0])] += 1
    return counts


def event_targets(times, events, *, counts):
    y, ids = assign_events(times, events)
    counts = np.asarray(counts)
    if counts.shape != (6,) or (counts < 0).any():
        raise ValueError('six nonnegative family counts required')
    weights = np.zeros(len(times), dtype=np.float32)
    for index in np.unique(ids[ids >= 0]):
        keep = ids == index
        family = int(y[keep][0])
        if counts[family] == 0:
            raise ValueError('represented family has no training count')
        weights[keep] = 1./(6*counts[family]*keep.sum())
    return dict(event_y=torch.as_tensor(y), event_weight=torch.as_tensor(weights))


def event_objective(logits, labels, weights):
    if (logits.shape != (len(labels), 6) or labels.ndim != 1 or weights.shape != labels.shape
            or ((labels < -1) | (labels > 5)).any() or (weights < 0).any()
            or not torch.isfinite(weights).all() or not torch.isfinite(logits).all()):
        raise ValueError('finite six-family logits, labels and nonnegative weights required')
    known = labels >= 0
    if (weights[~known] != 0).any():
        raise ValueError('unknown intervals must have zero weight')
    if not known.any():
        return logits.sum()*0
    return (F.cross_entropy(logits[known], labels[known], reduction='none')*weights[known]).sum()


def risk_and_features(model, h, valid, delta):
    """Tap the actual frozen classifier input; do not reimplement risk inference."""
    captured = []

    def capture(module, args):
        captured.append(args[0])  # no detach: auxiliary gradient must reach adaptation

    hook = model.base.risk_head.register_forward_pre_hook(capture)
    try:
        z = model.risk(h, valid, delta)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].shape != (len(h), model.base.cfg.dim):
        raise RuntimeError('exactly one classifier representation required')
    return z, captured[0]


def loss_from_example(model, head, e, threshold):
    trace = model.memory.trace(e['means'], e['complete'], mode='memory', return_before=True)
    delta = endpoint_corrections(trace['delta'], e['ends'])
    before = endpoint_corrections(trace['before_delta'], e['ends'])
    h, valid = e['h'], e['valid']
    after_logits, before_logits, representations = [], [], []
    for i in range(0, len(h), 32):
        z, features = risk_and_features(model, h[i:i+32], valid[i:i+32], delta[i:i+32])
        after_logits.append(z)
        representations.append(features)
        before_logits.append(model.risk(h[i:i+32], valid[i:i+32], before[i:i+32]))
    risk, info = objective(torch.cat(after_logits), torch.cat(before_logits), e, threshold, 'direction')
    logits = head(torch.cat(representations))
    event = event_objective(logits, e['event_y'], e['event_weight'])
    correct = (logits.detach().argmax(-1) == e['event_y']).to(logits)
    detail = {k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in info.items()}
    detail.update(event_loss=float(event.detach()),
                  event_correct_weight=float((correct*e['event_weight']).sum()),
                  event_weight=float(e['event_weight'].sum()),
                  event_points=int((e['event_y'] >= 0).sum()), writes=trace['updates'])
    return risk, event, detail
