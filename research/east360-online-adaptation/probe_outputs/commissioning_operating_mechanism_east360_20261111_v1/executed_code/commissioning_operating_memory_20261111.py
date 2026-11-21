"""Per-shot local decoder adaptation from previously matured forecast errors."""

import numpy as np

from .commissioning_weak_shape_data_20261111 import GROUPS


WINDOW, HISTORY, NEIGHBORS = 25, 12, 2
STRENGTH = np.float32(.5)
MODES = ('uncalibrated', 'matched', 'recent', 'wrong_context')
_FAMILY = np.asarray(GROUPS)
_TIME = np.linspace(-1., 1., WINDOW, dtype=np.float32)[:, None]


def _mean(values, mask):
    n = mask.sum(0)
    total = np.where(mask, values, 0.).sum(0)
    return np.divide(total, n, out=np.zeros_like(total), where=n > 0)


def innovation(residual, observed):
    """Level and centered slope of bounded diagnostics, in channel order."""
    level = _mean(residual, observed)
    time = _TIME - _mean(_TIME, observed)
    slope = _mean(np.clip(residual-level, -1., 1.)*time, observed)
    return np.column_stack((_mean(np.clip(residual, -1., 1.), observed), slope))


def _issue(reference, point, context, prior, errors, masks):
    forecasts = dict.fromkeys(MODES, reference)
    if not len(prior):
        return forecasts, False
    d = ((context[prior] - point)**2).sum(1)
    choices = dict(matched=prior[np.argsort(d, kind='stable')[:NEIGHBORS]],
                   recent=prior[:NEIGHBORS],
                   wrong_context=prior[np.argsort(-d, kind='stable')[:NEIGHBORS]])
    for k, ix in choices.items():
        count = masks[ix].sum(0)
        bias = np.divide(errors[ix].sum(0), count,
                         out=np.zeros_like(reference), where=count > 0)
        forecasts[k] = reference + STRENGTH*bias
    return forecasts, set(choices['matched']) != set(choices['recent'])


def replay(forecast, target, observed, origins, context, valid_context, length):
    """Issue all 25 predictions from one causal origin, then update after arrival.

    At issuance, only earlier blocks with a mature truth and usable context can
    be selected. Each nonempty neighborhood takes one bounded step from zero
    on its local Huber loss for the 25-by-10 decoder intercept. Equal numbers
    of prior blocks drive the matched, recent and mismatched-context controls.
    After q+25 arrives, the prediction innovation updates the hidden state
    starting at q+26. This function owns no persistent state or shot outcome.
    """
    if (forecast.shape != target.shape or forecast.shape != observed.shape
            or forecast.shape != (len(origins), WINDOW, len(_FAMILY))
            or context.shape != (len(origins), 3)
            or valid_context.shape != origins.shape or np.any(np.diff(origins) <= WINDOW)
            or np.any((origins < 0) | (origins >= length))):
        raise ValueError('ordered origins, non-overlapping complete blocks and aligned context required')
    states = {k: np.zeros((length, len(_FAMILY)*2), np.float32) for k in MODES}
    running = {k: np.zeros((len(_FAMILY), 2), np.float32) for k in MODES}
    sse = dict.fromkeys(MODES, 0.)
    n, released, changed = 0, 0, 0
    gradient = np.zeros_like(target)
    memory = []
    for j, q in enumerate(origins):
        start = q + WINDOW + 1
        if start >= length:
            break
        stop = origins[j+1]+WINDOW+1 if j+1 < len(origins) else length
        # At issuance, only completed, earlier blocks are passed to the learner.
        available = np.array(memory[-HISTORY:][::-1] if valid_context[j] else (), dtype=np.int64)
        issued, different = _issue(forecast[j], context[j], context, available, gradient, observed)
        # Future observations are used only at the q+25 reveal, not by _issue.
        m = observed[j]
        update = m.sum(0) >= (WINDOW+1)//2
        if np.unique(_FAMILY[update]).size >= 2:
            changed += different
            for k, prediction in issued.items():
                residual = np.where(m, target[j]-prediction, 0.)
                sse[k] += float(np.square(residual[m].astype(np.float64)).sum())
                step = innovation(residual, m)
                has = m.any(0)
                running[k][has] = .5*running[k][has] + .5*step[has]
            # Record this block only after all four issued forecasts were evaluated.
            if valid_context[j]:
                gradient[j] = np.where(m, np.clip(target[j]-forecast[j], -1., 1.), 0.)
                memory.append(j)
            n += int(m.sum())
            released += 1
        for k in MODES:
            states[k][start:stop] = running[k].ravel()
    return states, dict(forecast_sq=sse, observed_values=n, released=released,
                        matched_differs_from_recent=changed)
