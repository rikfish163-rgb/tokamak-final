"""Target-to-source maps in the frozen model's asinh input coordinates.

These are statistical comparators, not certified physical calibration. Missing
columns remain missing. Each inference availability pattern uses its observed
principal correlation matrices; no missing value is used as a measured zero.
Correlations are estimated using mean-imputed standardized calibration values,
then normalized to a unit diagonal. Missingness can still bias this estimator;
the masked-pattern restriction does not make that uncertainty disappear.
"""

import numpy as np

from ..io23 import RAW23

EXCLUDED = {"MNM", "mode_number_n"}


def ended_groups(rows, flags, *, order, count, seals):
    ids = [r["shot"] for r in rows]
    if (
        type(count) is not int
        or not 0 < count < len(order)
        or len(order) != len(set(order))
    ):
        raise ValueError("explicit ended prefix count required")
    if ids != list(order[:count]) or set(flags) != set(ids) or set(seals) != set(ids):
        raise ValueError("only exact sealed and revealed prefix allowed")
    if any(type(v) is not bool for v in flags.values()) or any(
        not isinstance(s, str)
        or len(s) != 64
        or any(c not in "0123456789abcdef" for c in s)
        for s in seals.values()
    ):
        raise ValueError("boolean ended outcomes and SHA256 prediction seals required")
    for row in rows:
        _signal(row, True)
    safe = [r for r in rows if not flags[r["shot"]]]
    if not safe:
        raise ValueError("no ended SAFE calibration; do not invent a safe reference")
    return dict(pooled=rows, safe=safe)


def _signal(row, with_id=False):
    if set(row) != (
        {"shot", "values", "available"} if with_id else {"values", "available"}
    ):
        raise ValueError("explicit signal-only record required")
    x, m = np.asarray(row["values"]), np.asarray(row["available"])
    if (
        x.ndim != 2
        or x.shape[1] != 23
        or not len(x)
        or x.dtype != np.float32
        or m.shape != x.shape
        or m.dtype != bool
    ):
        raise ValueError("nonempty float32 flat23 values and boolean mask required")
    if not np.isfinite(x[m]).all():
        raise ValueError("nonfinite observed signal")
    return x, m


def _normalizer(center, scale):
    center, scale = np.asarray(center, np.float64), np.asarray(scale, np.float64)
    if (
        center.shape != (23,)
        or scale.shape != (23,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("finite fixed source center/positive scale required")
    return center, scale


def fit_coordinates(rows, center, scale):
    center, scale = _normalizer(center, scale)
    if not rows:
        raise ValueError("nonempty calibration records required")
    ids = [r["shot"] for r in rows]
    if any(type(s) is not int for s in ids) or len(ids) != len(set(ids)):
        raise ValueError("distinct integer shot identities required")
    xs, ms, ws = [], [], []
    counts = np.zeros(23, int)
    for row in rows:
        x, m = _signal(row, True)
        take = np.linspace(0, len(x) - 1, min(128, len(x)), dtype=int)
        z = np.arcsinh((np.where(m[take], x[take], center) - center) / scale)
        xs.append(z)
        ms.append(m[take])
        ws.append(np.full(len(take), 1 / len(take)))
        counts += m[take].any(0)
    z, m, w = np.concatenate(xs), np.concatenate(ms), np.concatenate(ws)
    means = np.zeros(23)
    std = np.zeros(23)
    for c in range(23):
        usable = m[:, c]
        v = z[usable, c]
        weight = w[usable]
        if not len(v):
            continue
        weight = weight / weight.sum()
        means[c] = weight @ v
        if v.min() != v.max():
            std[c] = np.sqrt(weight @ np.square(v - means[c]))
    active = std > 0
    normalized = np.zeros_like(z)
    normalized[:, active] = np.where(
        m[:, active], (z[:, active] - means[active]) / std[active], 0.0
    )
    cov = (normalized.T * w) @ normalized / w.sum()
    denominator = np.sqrt(np.diag(cov))
    corr = np.eye(23)
    selected = np.flatnonzero(active)
    if len(selected):
        ix = np.ix_(selected, selected)
        corr[ix] = cov[ix] / np.outer(denominator[selected], denominator[selected])
    if not np.isfinite(corr).all():
        raise FloatingPointError("nonfinite correlation")
    return dict(
        shot_ids=ids,
        mean=means.tolist(),
        std=std.tolist(),
        correlation=corr.tolist(),
        observed_shots=counts.tolist(),
        sample_cap_per_shot=128,
        covariance_missing="mean_imputation_then_unit_diagonal",
        independent_sample_unit="shot; sampled timepoints are correlated",
    )


def covariance_matrix(source, target):
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if (
        source.ndim != 2
        or source.shape != target.shape
        or source.shape[0] != source.shape[1]
        or not len(source)
    ):
        raise ValueError("matching nonempty square covariances required")

    def power(c, exponent):
        if not np.isfinite(c).all() or not np.allclose(c, c.T, atol=1e-12):
            raise ValueError("finite symmetric covariance required")
        eigen, u = np.linalg.eigh(c + np.eye(len(c)))
        if (eigen <= 0).any():
            raise ValueError("ridge-regularized covariance not positive definite")
        return (u * eigen**exponent) @ u.T

    # Row vectors: A.T (Ct+I) A = Cs+I. This is TARGET -> SOURCE.
    a = power(target, -0.5) @ power(source, 0.5)
    if not np.allclose(
        a.T @ (target + np.eye(len(a))) @ a,
        source + np.eye(len(a)),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise AssertionError("target-to-source covariance identity failed")
    return a


def make_transport(source, target, mode):
    if mode not in ("diagonal", "correlation"):
        raise ValueError("explicit alignment mode required")
    active = []
    status = []
    for c, name in enumerate(RAW23):
        if name in EXCLUDED:
            why = "excluded_semantic_uncertainty"
        elif min(source["observed_shots"][c], target["observed_shots"][c]) < 3:
            why = "fewer_than_three_observed_shots"
        elif min(source["std"][c], target["std"][c]) <= 0:
            why = "unidentifiable_zero_variation"
        else:
            why = "mapped"
        active.append(why == "mapped")
        status.append(why)
    return dict(
        mode=mode,
        source=source,
        target=target,
        active=active,
        status=status,
        ridge=1.0,
        direction="target_to_frozen_source",
        physical_calibration_certified=False,
    )


def transport(signal, fit, center, scale):
    x, m = _signal(signal)
    center, scale = _normalizer(center, scale)
    out = x.copy()
    active = np.asarray(fit["active"], bool)
    z = np.arcsinh((np.where(m, x, center) - center) / scale)
    sm, ss = np.asarray(fit["source"]["mean"]), np.asarray(fit["source"]["std"])
    tm, ts = np.asarray(fit["target"]["mean"]), np.asarray(fit["target"]["std"])
    cs, ct = (
        np.asarray(fit["source"]["correlation"]),
        np.asarray(fit["target"]["correlation"]),
    )
    patterns, inverse = np.unique(m & active, axis=0, return_inverse=True)
    for pattern_index, pattern in enumerate(patterns):
        cols = np.flatnonzero(pattern)
        if not len(cols):
            continue  # Explicitly no observed identifiable coordinate to transform.
        points = np.flatnonzero(inverse == pattern_index)
        ix = np.ix_(points, cols)
        standard = (z[ix] - tm[cols]) / ts[cols]
        if fit["mode"] == "correlation":
            block = np.ix_(cols, cols)
            standard = standard @ covariance_matrix(cs[block], ct[block])
        elif fit["mode"] != "diagonal":
            raise ValueError("unknown map mode")
        mapped_z = standard * ss[cols] + sm[cols]
        with np.errstate(over="raise", invalid="raise"):
            mapped = center[cols] + scale[cols] * np.sinh(mapped_z)
        if (
            not np.isfinite(mapped).all()
            or (abs(mapped) > np.finfo(np.float32).max).any()
        ):
            raise FloatingPointError(
                "transport outside representable raw range; no silent clipping"
            )
        out[ix] = mapped.astype(np.float32)
    return dict(values=out, available=m.copy())
