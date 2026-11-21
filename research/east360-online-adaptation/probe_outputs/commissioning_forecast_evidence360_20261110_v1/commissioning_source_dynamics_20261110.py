"""Source-only direct forecasts for matched activity in safe and disruptive shots."""

import numpy as np

from . import io23, simple_data
from .commissioning_event_pairs_20261110 import match
from .commissioning_event_roughness_20261110 import COLUMNS, CONTEXT, prefix_context
from .commissioning_roughness_20261110 import coordinates
from .event_prefix_probe import LABEL_SHA256, assign_events
from .source_event_tasks_20261102.data import load_events


HORIZON = 25


def selected_origins(q, times, family, ready, disruptive, down):
    """Use named activities only; disruptive targets end strictly before lead 30 ms."""
    label = family[q]
    eligible = np.asarray(ready, bool) & (label >= 0)
    if not disruptive:
        return np.flatnonzero(eligible)
    lead = np.round((down - times[q]) * 1000., 9)
    mature_lead = np.round((down - times[q+HORIZON]) * 1000., 9)
    eligible &= (lead <= 125.) & (mature_lead > 30.)
    chosen = []
    for f in np.unique(label[eligible]):
        indices = np.flatnonzero(eligible & (label == f))
        chosen.append(indices[np.abs(lead[indices] - 90.).argmin()])
    return np.array(chosen, dtype=int)


def shot_blocks(signal, center, scale):
    from .commissioning_dual_dynamics_20261110 import prepare

    indices = COLUMNS + CONTEXT
    z, m = coordinates(signal['values'][:, indices], signal['available'][:, indices],
                       center[indices], scale[indices])
    return prepare(z[:, :-2], m[:, :-2], z[:, -2:], m[:, -2:]), prefix_context(z[:, -2:], m[:, -2:])


def candidates(center, scale):
    events, _ = load_events()
    training = io23.source_roles()['train']
    by_outcome = {False: [], True: []}
    for sid in training:
        if sid not in events:
            continue
        d, down = io23.read_outcome('J-TEXT', sid)
        s = simple_data.read_shot('J-TEXT', sid)
        (q, x, y, m), (ctx, ctx_ok) = shot_blocks(s, center, scale)
        family, _ = assign_events(s['times'], events[sid])
        ready = ctx_ok[q] & ((m.sum(axis=1) >= 12).sum(axis=1) >= 6)
        for j in selected_origins(q, s['times'], family, ready, d, down):
            by_outcome[d].append(dict(shot=sid, family=int(family[q[j]]), endpoint=int(q[j]),
                                      context=ctx[q[j]], feature=x[j], y=y[j], m=m[j]))
    return by_outcome


def fit_pair_models(pairs):
    from .commissioning_dual_dynamics_20261110 import fit

    if not pairs:
        raise ValueError('no same-activity training pairs')
    family = np.array([d['family'] for d, _ in pairs])
    _, inverse, count = np.unique(family, return_inverse=True, return_counts=True)
    w = 1. / (len(count) * count[inverse])
    result = []
    for group in (1, 0):  # benign first
        x, y, m = (np.stack([pair[group][k] for pair in pairs]) for k in ('feature', 'y', 'm'))
        result.append(fit(x, y, m, w))
    return tuple(result)


def held_statistics(pairs, models):
    means, advantages = [], []
    for group in (0, 1):
        x, y, m = (np.stack([pair[group][k] for pair in pairs]) for k in ('feature', 'y', 'm'))
        errors = []
        for pred in (np.zeros_like(y), *(np.einsum('bd,dhc->bhc', x, p) for p in models)):
            a = np.abs(pred-y)
            b = np.minimum(a, .5)
            errors.append(((.5*b*b + .5*(a-b)) * m).sum((1, 2)) / m.sum((1, 2)))
        means.append(dict(zip(('hold_recent_mean', 'benign', 'disruptive'),
                              (float(np.mean(v)) for v in errors), strict=True)))
        advantages.append(errors[1] - errors[2])
    delta = advantages[0] - advantages[1]
    return dict(disruptive_error=means[0], safe_error=means[1],
                paired_preference=float(np.mean((delta > 0) + .5*(delta == 0))),
                correct_pairs=int((delta > 0).sum()), ties=int((delta == 0).sum()))


def fit_source(center, scale):
    by_outcome = candidates(center, scale)
    positive, safe = by_outcome[True], by_outcome[False]
    folds = {s: i % 4 for rows in by_outcome.values()
             for i, s in enumerate(sorted({r['shot'] for r in rows}))}
    held = []
    for f in range(4):
        train = match(*([r for r in g if folds[r['shot']] != f] for g in (positive, safe)))
        test = match(*([r for r in g if folds[r['shot']] == f] for g in (positive, safe)))
        train_ids = sorted({r['shot'] for pair in train for r in pair})
        held_ids = sorted({r['shot'] for pair in test for r in pair})
        if not test or set(train_ids) & set(held_ids):
            raise ValueError('nonempty, shot-disjoint source folds required')
        held.append(dict(fold=f, train_pairs=len(train), held_pairs=len(test),
                         train_shots=train_ids, held_shots=held_ids,
                         **held_statistics(test, fit_pair_models(train))))
    pairs = match(positive, safe)
    return fit_pair_models(pairs), dict(
        scope='SOURCE_TRAIN_ONLY_EVENT_ACTIVITY_MATCHED',
        event_csv_sha256=LABEL_SHA256, candidate_blocks={'safe': len(safe), 'disruptive': len(positive)},
        paired_shots={'safe': len({s['shot'] for _, s in pairs}),
                      'disruptive': len({d['shot'] for d, _ in pairs})},
        pairs=[[[r['shot'], r['family'], r['endpoint']] for r in pair] for pair in pairs],
        held_source=held)
