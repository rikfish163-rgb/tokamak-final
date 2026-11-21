"""TRAIN-only coordinates/events; causal feedback pairs without outcome inputs."""
from dataclasses import replace

import numpy as np

from .. import io23, simple_data
from ..event_prefix_probe import assign_events


def _require_train(shots, allowed_train_ids):
    allowed = set(allowed_train_ids)
    if not shots or any(s['device'] != 'J-TEXT' or s['shot'] not in allowed for s in shots):
        raise ValueError('only declared J-TEXT TRAIN shots may fit coordinates or event targets')


def attach_training_events(shots, grouped, *, allowed_train_ids, shuffle_seed=None):
    _require_train(shots, allowed_train_ids)
    selected = {s['shot']: list(grouped.get(s['shot'], [])) for s in shots}
    if shuffle_seed is not None:
        # Whole-event shuffle within outcome and source-ID quartile, never
        # moving event timing or borrowing a validation/test event label.
        rng = np.random.default_rng(shuffle_seed)
        bounds = np.quantile([s['shot'] for s in shots], [.25, .5, .75])
        strata = {}
        for s in shots:
            key = (bool(s['is_disrupt']), int(np.searchsorted(bounds, s['shot'])))
            for i, e in enumerate(selected[s['shot']]):
                strata.setdefault(key, []).append((s['shot'], i, e))
        for rows in strata.values():
            labels = rng.permutation([e.family for _, _, e in rows])
            for (sid, i, e), label in zip(rows, labels):
                selected[sid][i] = replace(e, family=int(label))
    return [dict(s, event_target=assign_events(s['times'], selected[s['shot']])[0],
                 event_intervals=selected[s['shot']]) for s in shots]


def fit_coordinates(shots, *, allowed_train_ids, horizon):
    _require_train(shots, allowed_train_ids)
    if horizon < 1:
        raise ValueError('positive direct forecast horizon required')
    centers, scales, variable = [], [], []
    for c in range(23):
        samples = [s['values'][s['available'][:, c] & s['valid'], c] for s in shots]
        v = np.concatenate(samples).astype('float64')
        if not len(v) or not np.isfinite(v).all():
            raise ValueError(f'no finite source TRAIN data for {io23.RAW23[c]}')
        q25, center, q75 = np.quantile(v, [.25, .5, .75])
        # Sparse/non-Gaussian channels can have IQR=0 without being constant.
        # Source SD supplies a declared scale floor; a true constant uses 1
        # and is excluded from continuous forecasting, rather than /1e-6.
        varies = bool(v.max() > v.min())
        scale = max(q75-q25, .1*np.std(v), .01*abs(center)) if varies else 1.
        centers.append(center); scales.append(scale); variable.append(varies)
    center, scale = np.asarray(centers, 'float32'), np.asarray(scales, 'float32')
    sums, counts = np.zeros(23, 'float64'), np.zeros(23, 'int64')
    for s in shots:
        z = np.arcsinh((s['values']-center)/scale)
        for h in sorted(set([1, max(1, horizon//4), max(1, horizon//2), horizon])):
            m = s['available'][h:] & s['available'][:-h] & s['valid'][h:, None] & s['valid'][:-h, None]
            d = np.where(m, z[h:]-z[:-h], 0.).astype('float64')
            sums += (d*d).sum(0); counts += m.sum(0)
    q = np.sqrt(sums/np.maximum(counts, 1))
    active = np.asarray(variable) & (q > 0)
    # Mode index is not a continuous diagnostic amplitude; retain its input,
    # but do not pretend squared/Huber distance between mode IDs is physics.
    active[io23.RAW23.index('mode_number_n')] = False
    q = np.where(active, q, 1.).astype('float32')
    return dict(center=center, scale=scale, change_scale=q, forecast_active=active)


def feedback_sample(s, *, origin, now, cfg):
    if not 0 <= origin <= now < len(s['values']):
        raise ValueError('feedback requires origin <= observed now within this shot')
    x, mask = simple_data.window(s, origin, cfg.window)
    target = np.zeros((cfg.horizon, 23), 'float32')
    tm = np.zeros_like(target, bool)
    # Slice ends at NOW, not the preloaded record end. No labels are consulted.
    n = min(cfg.horizon, now-origin)
    if n:
        target[:n] = s['values'][origin+1:origin+1+n]
        tm[:n] = s['available'][origin+1:origin+1+n]
    return dict(x=x, mask=mask, target=target, target_mask=tm)
