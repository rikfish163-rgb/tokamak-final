"""Small constrained quadratic fit, independent of HDF5 or neural inference."""

from collections import Counter, defaultdict

import numpy as np


def pair(row):
    s = np.asarray(
        row["results"]["difference_real"]["receipt"]["gradient256"], np.float64
    )
    r = np.asarray(row["task_gradient256"], np.float64)
    if (
        s.ndim != 1
        or not len(s)
        or r.shape != s.shape
        or not np.isfinite(s).all()
        or not np.isfinite(r).all()
    ):
        raise ValueError("matching finite nonempty gradient vectors required")
    ns, nr = np.linalg.norm(s), np.linalg.norm(r)
    if not np.isfinite(ns) or not np.isfinite(nr):
        raise FloatingPointError("gradient norm overflow")
    if ns == 0 or nr == 0:
        return None
    return s / ns, r / nr


def row_weights(rows):
    counts = Counter((row["label"], row["shot"]) for row in rows)
    if not rows or any(
        type(row["label"]) is not int or row["label"] not in (0, 1) for row in rows
    ):
        raise ValueError("binary label groups required")
    shots = {y: {row["shot"] for row in rows if row["label"] == y} for y in (0, 1)}
    if not all(shots.values()):
        raise ValueError("both label groups required for a balanced fit")
    return np.asarray(
        [
            0.5 / len(shots[row["label"]]) / counts[(row["label"], row["shot"])]
            for row in rows
        ]
    )


def coefficients(signal, task, weights):
    s, r, w = (
        np.asarray(signal, np.float64),
        np.asarray(task, np.float64),
        np.asarray(weights, np.float64),
    )
    if (
        s.ndim != 2
        or not s.shape[0]
        or not s.shape[1]
        or r.shape != s.shape
        or w.shape != s.shape[:1]
        or not all(np.isfinite(v).all() for v in (s, r, w))
        or (w < 0).any()
        or not np.isclose(w.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError(
            "finite gradient matrices and normalized nonnegative sample weights required"
        )
    prior = 1.0 / s.shape[1]
    return w @ (s * s) + prior, w @ (s * r) + prior


def solve(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if (
        a.ndim != 1
        or not len(a)
        or b.shape != a.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or (a <= 0).any()
    ):
        raise ValueError(
            "finite equal coefficient vectors with positive quadratic terms required"
        )
    with np.errstate(over="raise", invalid="raise"):
        lo, hi = float(np.min(b - 4 * a)), float(np.max(b))
        for _ in range(80):
            mu = lo + (hi - lo) / 2
            if np.clip((b - mu) / a, 0, 4).sum() > len(a):
                lo = mu
            else:
                hi = mu
        mu = lo + (hi - lo) / 2
        p = np.clip((b - mu) / a, 0, 4)
    if not np.isfinite(p).all() or abs(p.sum() - len(a)) > 1e-10:
        raise FloatingPointError("quadratic solution failed its feasibility check")
    return p, mu


def fit(rows):
    if len({(row["shot"], row["query"]) for row in rows}) != len(rows):
        raise ValueError("duplicate training query")
    normalized = [pair(row) for row in rows]
    chosen = [
        row for row, values in zip(rows, normalized, strict=True) if values is not None
    ]
    selected = [values for values in normalized if values is not None]
    if not selected:
        raise ValueError("no defined gradient pairs")
    s, r = np.stack([v[0] for v in selected]), np.stack([v[1] for v in selected])
    w = row_weights(chosen)
    a, b = coefficients(s, r, w)
    p, mu = solve(a, b)
    objective = float(w @ np.sum((s * p - r) ** 2, 1) + np.mean((p - 1) ** 2))
    identity = float(w @ np.sum((s - r) ** 2, 1))
    if objective > identity + 1e-12:
        raise FloatingPointError("fit is worse than the feasible identity metric")
    return dict(
        weights=p.tolist(),
        multiplier=mu,
        quadratic=a.tolist(),
        linear=b.tolist(),
        n_total=len(rows),
        n_defined=len(chosen),
        training_ids=sorted({row["shot"] for row in rows}),
        undefined_pairs=[
            {key: row[key] for key in ("shot", "query")}
            for row, values in zip(rows, normalized, strict=True)
            if values is None
        ],
        objective_fit=objective,
        objective_identity=identity,
    )


def evaluate(row, weights):
    p = np.asarray(weights, np.float64)
    values = pair(row)
    result = {key: row[key] for key in ("shot", "query", "label", "disruptive")}
    if values is None:
        return dict(
            **result,
            before_cosine=None,
            after_cosine=None,
            first_order_signal_descent=None,
        )
    s, r = values
    if p.shape != s.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 4).any():
        raise ValueError("matching feasible coordinate weights required")
    changed = p * s
    norm = np.linalg.norm(changed)
    return dict(
        **result,
        before_cosine=float(np.clip(s @ r, -1, 1)),
        after_cosine=float(np.clip(changed @ r / norm, -1, 1)) if norm else None,
        first_order_signal_descent=float(s @ changed),
    )


def group_summary(rows):
    result = {}
    for name in ("class0", "class1", "safe"):
        if name == "safe":
            group = [r for r in rows if not r["disruptive"] and r["label"] == 0]
        else:
            group = [r for r in rows if r["label"] == int(name[-1])]
        data = dict(n_total=len(group), n_shots=len({r["shot"] for r in group}))
        for phase in ("before", "after"):
            by_shot = defaultdict(list)
            for row in group:
                value = row[phase + "_cosine"]
                if value is not None:
                    by_shot[row["shot"]].append(value)
            data[phase + "_defined"] = sum(map(len, by_shot.values()))
            data[phase + "_mean_cosine"] = (
                float(np.mean([np.mean(v) for v in by_shot.values()]))
                if by_shot
                else None
            )
            data[phase + "_positive_fraction"] = (
                float(np.mean([np.mean(np.asarray(v) > 0) for v in by_shot.values()]))
                if by_shot
                else None
            )
        before, after = data["before_mean_cosine"], data["after_mean_cosine"]
        data["gain"] = (
            after - before if before is not None and after is not None else None
        )
        result[name] = data
    coverage = all(
        g["n_total"] > 0 and g["before_defined"] == g["after_defined"] == g["n_total"]
        for g in result.values()
    )
    gates = dict(coverage=coverage)
    for name in ("class0", "class1"):
        group = result[name]
        gates[name] = (
            group["gain"] is not None
            and group["after_mean_cosine"] > 0
            and group["gain"] >= 0.01
        )
    safe = result["safe"]
    gates["safe"] = (
        safe["gain"] is not None and safe["after_mean_cosine"] > 0 and safe["gain"] >= 0
    )
    gates["passed"] = all(gates.values())
    return dict(groups=result, gates=gates)
