"""Fit a fixed first-layer write selector with shot-level historical supervision."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
import torch


TEMPERATURE, RIDGE, FIRST, STRIDE = .20, .05, 135, 32


def logit(prob):
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1. - 1e-7)
    return np.log(p) - np.log1p(-p)


def no_alarm_log(gate, survival, *, reduction=True):
    mixed = (1. - gate) * survival[:, 0] + gate * survival[:, 1]
    z = mixed.log()
    return z.sum() if reduction else z


def positive_probability(gate, early, until, prefix):
    a = prefix[0] * no_alarm_log(gate, early).exp()
    b = prefix[1] * no_alarm_log(gate, until).exp()
    return (a - b).clamp_min(0.)


@dataclass(frozen=True)
class Teacher:
    until: np.ndarray
    early: np.ndarray | None
    prefix: np.ndarray | None


def make_teacher(times, static, adapted, down, tau, *, first=FIRST, stride=STRIDE):
    """Soft survival for each hypothetical write, on already revealed history."""
    scores = logit(np.column_stack((static, adapted)))
    start = np.arange(first, len(times), stride)
    if not len(start):
        raise ValueError("no historical first-layer actions")
    boundary = float(logit(tau))

    def survival(mask):
        z = np.where(mask[:, None], scores, -np.inf)
        peak = np.maximum.reduceat(z, start, axis=0)
        prefix_peak = z[:first, 0].max(initial=-np.inf)
        return expit((boundary - peak) / TEMPERATURE), float(expit((boundary-prefix_peak)/TEMPERATURE))

    if down is None:
        return Teacher(survival(np.ones(len(times), dtype=bool))[0], None, None)
    lead = np.round((down - times) * 1000., 9)
    early, a = survival(lead > 3000.)
    until, b = survival(lead > 30.)
    return Teacher(until, early, np.array([a, b]))


def safe_weights(survival, scope):
    n = len(survival)
    if scope == "rolling":
        # Positive uniform micro-weight also covers numerically zero-risk blocks.
        weights = 1. - survival.min(axis=1) + 1e-8
    else:
        weights = np.zeros(n)
        if scope == "peak1":
            bins = 1
        elif scope == "peak8":
            bins = min(8, n)
        else:
            raise ValueError(f"unknown training scope: {scope}")
        for ids in np.array_split(np.arange(n), bins):
            weights[ids[survival[ids, 0].argmin()]] = 1.
    return weights / weights.sum()


def features(times, static, history, *, first=FIRST, stride=STRIDE):
    ix = np.arange(first, len(times), stride)
    if history.shape != (len(ix), 128):
        raise ValueError("one causal history per release")
    p = static.astype(np.float64)
    basic = np.column_stack((logit(p[ix]), np.log1p(times[ix]-times[0]),
                             p[ix] - p[np.maximum(0, ix-stride)]))
    return np.column_stack((basic, history))


@dataclass(frozen=True)
class Gate:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray

    def score(self, observed):
        x = (observed[:, :len(self.mean)] - self.mean) / self.scale
        return self.weight[0] + x @ self.weight[1:]

    def select(self, observed):
        return self.score(observed) > 0.


def fit(shots, scope, *, feedback=True):
    """One deterministic full-batch fit; never learns from current labels."""
    width = 131 if feedback else 3
    values = [x[:, :width] for x, _ in shots]
    mean = np.mean([x.mean(axis=0) for x in values], axis=0)
    var = np.mean([np.square(x-mean).mean(axis=0) for x in values], axis=0)
    scale = np.sqrt(np.maximum(var, 1e-4))
    negatives, positives = [], []
    for x, (_, teacher) in zip(values, shots, strict=True):
        design = torch.from_numpy(np.column_stack((np.ones(len(x)), (x-mean)/scale)))
        q = torch.from_numpy(teacher.until)
        if teacher.early is None:
            negatives.append((design, q, torch.from_numpy(safe_weights(teacher.until, scope))))
        else:
            positives.append((design, torch.from_numpy(teacher.early), q,
                              torch.from_numpy(teacher.prefix)))
    if not positives or not negatives:
        raise ValueError("both observed history classes required")
    weight = torch.zeros(width + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([weight], max_iter=60, line_search_fn="strong_wolfe")

    def objective():
        pos = torch.stack([
            -positive_probability((x@weight).sigmoid(), e, u, p).clamp_min(1e-12).log()
            for x, e, u, p in positives
        ]).mean()
        neg = torch.stack([
            -(w * no_alarm_log((x@weight).sigmoid(), q, reduction=False)).sum()
            for x, q, w in negatives
        ]).mean()
        return .5 * (pos + neg) + .5 * RIDGE * weight.square().sum(), pos, neg

    def closure():
        opt.zero_grad()
        loss, _, _ = objective()
        loss.backward()
        return loss

    start = float(objective()[0].detach())
    opt.step(closure)
    end, pos, neg = (float(v.detach()) for v in objective())
    if not np.isfinite(end) or end > start:
        raise FloatingPointError("historical fit did not converge to a finite improving objective")
    return Gate(mean, scale, weight.detach().numpy().copy()), dict(
        before=start, after=end, positive=pos, safe=neg,
        iterations=int(opt.state[weight]["n_iter"]),
    )


def splice(static, adapted, write, *, first=FIRST, stride=STRIDE):
    indices = np.arange(first, len(static))
    if len(write) != len(np.arange(first, len(static), stride)) or static.shape != adapted.shape:
        raise ValueError("selection and archive clocks do not match")
    trace = static.copy()
    pick = indices[write[(indices-first)//stride]]
    trace[pick] = adapted[pick]
    return trace


def scalar_control(static, adapted, *, first=FIRST, stride=STRIDE):
    indices = np.arange(first, len(static))
    anchor = first + ((indices-first)//stride)*stride
    result = static.astype(np.float64, copy=True)
    a, s = logit(adapted), logit(static)
    result[indices] = expit(s[indices] + (a[anchor] - s[anchor]))
    # The anchor used for matching is already available at the write clock.
    result[first::stride] = adapted[first::stride]
    return result
