"""Two bounded channel-weight tables with a causal observed condition."""

import numpy as np
from scipy.optimize import minimize

from ..history_gradient_metric_20261102.core import row_weights, solve as project_one


def solve(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if (
        b.ndim != 1
        or not len(b)
        or len(b) % 2
        or a.shape != (len(b), len(b))
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.allclose(a, a.T, atol=1e-10, rtol=0)
        or np.linalg.eigvalsh(a).min() <= 0
    ):
        raise ValueError("finite symmetric positive-definite 2C quadratic required")
    a = (a + a.T) / 2
    c = len(b) // 2
    equality = np.zeros((2, 2 * c))
    equality[0, :c], equality[1, c:] = 1, 1
    result = minimize(
        lambda w: float(w @ a @ w - 2 * b @ w),
        np.ones(2 * c),
        jac=lambda w: 2 * (a @ w - b),
        bounds=[(0.0, 4.0)] * (2 * c),
        constraints=dict(
            type="eq", fun=lambda w: equality @ w - c, jac=lambda w: equality
        ),
        method="SLSQP",
        options=dict(ftol=1e-12, maxiter=500),
    )
    if not result.success:
        raise FloatingPointError(f"two-table QP failed: {result.message}")
    w = np.asarray(result.x, np.float64)
    v = w - 2 * (a @ w - b)
    projected = np.concatenate(
        [project_one(np.ones(c), block)[0] for block in (v[:c], v[c:])]
    )
    residual = float(np.linalg.norm(w - projected))
    if (
        (w < 0).any()
        or (w > 4).any()
        or np.max(np.abs(equality @ w - c)) > 1e-8
        or residual > 1e-5
        or result.fun > float(np.ones(2 * c) @ a @ np.ones(2 * c) - 2 * b.sum()) + 1e-9
    ):
        raise FloatingPointError("two-table QP failed optimality/feasibility")
    return w, dict(
        success=True, projected_gradient_residual=residual, iterations=int(result.nit)
    )


def fit(gradients, risk, rows, observations, condition):
    g, r, obs, u = (
        np.asarray(v, np.float64) for v in (gradients, risk, observations, condition)
    )
    if (
        g.ndim != 3
        or not all(g.shape)
        or r.shape != (g.shape[0], g.shape[2])
        or obs.shape != g.shape[:2]
        or u.shape != (g.shape[0],)
        or len(rows) != len(g)
        or not all(np.isfinite(v).all() for v in (g, r, obs, u))
        or (obs < 0).any()
        or (u < 0).any()
        or (u > 1).any()
    ):
        raise ValueError(
            "matching gradients, observations and finite probability conditions required"
        )
    if np.any(g[obs == 0] != 0):
        raise ValueError("unobserved channel has nonzero gradient")
    if len({(row["shot"], row["query"]) for row in rows}) != len(rows):
        raise ValueError("duplicate query")
    active = np.flatnonzero(obs.sum(0) > 0)
    if not len(active):
        raise ValueError("no training-observed channels")
    ns, nr = np.linalg.norm(g.sum(1), axis=1), np.linalg.norm(r, axis=1)
    if not np.isfinite(ns).all() or not np.isfinite(nr).all():
        raise FloatingPointError("norm overflow")
    defined = (ns > 0) & (nr > 0)
    selected = [row for row, flag in zip(rows, defined, strict=True) if flag]
    if not selected:
        raise ValueError("no defined training pairs")
    normalized = g[defined][:, active] / ns[defined, None, None]
    u = u[defined, None, None]
    features = np.concatenate(((1 - u) * normalized, u * normalized), axis=1)
    target = r[defined] / nr[defined, None]
    population = row_weights(selected)
    c = len(active)
    a = np.einsum("n,ncd,ned->ce", population, features, features) + np.eye(2 * c) / (
        2 * c
    )
    b = np.einsum("n,ncd,nd->c", population, features, target) + 1 / (2 * c)
    w, receipt = solve(a, b)
    weights = np.zeros((2, g.shape[1]))
    weights[:, active] = w.reshape(2, c)
    objective = float(
        population @ np.sum((np.einsum("c,ncd->nd", w, features) - target) ** 2, axis=1)
        + np.mean((w - 1) ** 2)
    )
    identity = float(population @ np.sum((normalized.sum(1) - target) ** 2, axis=1))
    if objective > identity + 1e-9:
        raise FloatingPointError("fit worse than identity")
    return dict(
        weights=weights.tolist(),
        active_indices=active.tolist(),
        quadratic=a.tolist(),
        linear=b.tolist(),
        training_ids=sorted({row["shot"] for row in rows}),
        n_total=len(rows),
        n_defined=len(selected),
        undefined_pairs=[
            {k: row[k] for k in ("shot", "query")}
            for row, flag in zip(rows, defined, strict=True)
            if not flag
        ],
        objective_fit=objective,
        objective_identity=identity,
        **receipt,
    )


def effective_weights(fitted, condition):
    u = float(condition)
    weights = np.asarray(fitted["weights"], np.float64)
    if not np.isfinite(u) or not 0 <= u <= 1:
        raise ValueError("condition must be a finite probability")
    if weights.ndim != 2 or weights.shape[0] != 2 or not np.isfinite(weights).all():
        raise ValueError("two finite weight tables required")
    if (weights < 0).any() or (weights > 4).any():
        raise ValueError("weight tables outside fixed bounds")
    return (1 - u) * weights[0] + u * weights[1]
