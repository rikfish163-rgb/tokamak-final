"""Nonnegative diagnostic-loss weights, fitted from ended-shot gradients."""

import numpy as np
from scipy.optimize import minimize

from ..history_gradient_metric_20261102.core import row_weights, solve as diagonal_solve


def solve(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if (
        a.ndim != 2
        or a.shape[0] == 0
        or a.shape[0] != a.shape[1]
        or b.shape != a.shape[:1]
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.allclose(a, a.T, atol=1e-10, rtol=0)
        or np.linalg.eigvalsh(a).min() <= 0
    ):
        raise ValueError(
            "finite symmetric positive-definite quadratic and matching linear term required"
        )
    # Only the symmetric part contributes to w.T @ A @ w; this removes
    # floating-order antisymmetry from an otherwise symmetric Gram matrix.
    a = (a + a.T) / 2
    c = len(b)
    result = minimize(
        lambda w: float(w @ a @ w - 2 * b @ w),
        np.ones(c),
        jac=lambda w: 2 * (a @ w - b),
        bounds=[(0.0, 4.0)] * c,
        constraints=dict(
            type="eq", fun=lambda w: w.sum() - c, jac=lambda w: np.ones_like(w)
        ),
        method="SLSQP",
        options=dict(ftol=1e-12, maxiter=500),
    )
    if not result.success:
        raise FloatingPointError(f"channel QP did not converge: {result.message}")
    w = np.asarray(result.x, np.float64)
    projected, _ = diagonal_solve(np.ones(c), w - 2 * (a @ w - b))
    residual = float(np.linalg.norm(w - projected))
    if (
        (w < 0).any()
        or (w > 4).any()
        or abs(w.sum() - c) > 1e-8
        or residual > 1e-5
        or result.fun > float(np.ones(c) @ a @ np.ones(c) - 2 * b.sum()) + 1e-9
    ):
        raise FloatingPointError("channel QP failed feasibility/optimality checks")
    return w, dict(
        success=True, projected_gradient_residual=residual, iterations=int(result.nit)
    )


def fit(gradients, risk, rows, observations):
    g, r, obs = (
        np.asarray(value, np.float64) for value in (gradients, risk, observations)
    )
    if (
        g.ndim != 3
        or not all(g.shape)
        or r.shape != (g.shape[0], g.shape[2])
        or obs.shape != g.shape[:2]
        or len(rows) != len(g)
        or not all(np.isfinite(v).all() for v in (g, r, obs))
        or (obs < 0).any()
    ):
        raise ValueError(
            "matching finite channel/risk gradients, observations and rows required"
        )
    if np.any(g[obs == 0] != 0):
        raise ValueError("unobserved channel has a nonzero gradient")
    if len({(row["shot"], row["query"]) for row in rows}) != len(rows):
        raise ValueError("duplicate training query")
    active = np.flatnonzero(obs.sum(0) > 0)
    if not len(active):
        raise ValueError("no observed training channels")
    ns, nr = np.linalg.norm(g.sum(1), axis=1), np.linalg.norm(r, axis=1)
    if not np.isfinite(ns).all() or not np.isfinite(nr).all():
        raise FloatingPointError("normalization overflow")
    defined = (ns > 0) & (nr > 0)
    selected = [row for row, flag in zip(rows, defined, strict=True) if flag]
    if not selected:
        raise ValueError("no defined training direction pairs")
    s = g[defined][:, active] / ns[defined, None, None]
    target = r[defined] / nr[defined, None]
    sample_weights = row_weights(selected)
    c = len(active)
    a = np.einsum("n,ncd,ned->ce", sample_weights, s, s) + np.eye(c) / c
    b = np.einsum("n,ncd,nd->c", sample_weights, s, target) + 1.0 / c
    w, receipt = solve(a, b)
    weights = np.zeros(g.shape[1])
    weights[active] = w
    objective = float(
        sample_weights @ np.sum((np.einsum("c,ncd->nd", w, s) - target) ** 2, 1)
        + np.mean((w - 1) ** 2)
    )
    identity = float(sample_weights @ np.sum((s.sum(1) - target) ** 2, 1))
    if objective > identity + 1e-9:
        raise FloatingPointError("channel metric is worse than equal channel losses")
    return dict(
        weights=weights.tolist(),
        active_indices=active.tolist(),
        quadratic=a.tolist(),
        linear=b.tolist(),
        training_ids=sorted({row["shot"] for row in rows}),
        n_total=len(rows),
        n_defined=len(selected),
        undefined_pairs=[
            {key: row[key] for key in ("shot", "query")}
            for row, flag in zip(rows, defined, strict=True)
            if not flag
        ],
        objective_fit=objective,
        objective_identity=identity,
        **receipt,
    )


def evaluate(gradients, risk, row, weights):
    g, r, w = (np.asarray(value, np.float64) for value in (gradients, risk, weights))
    original, changed = g.sum(0), w @ g
    ns, nc, nr = np.linalg.norm(original), np.linalg.norm(changed), np.linalg.norm(r)
    return dict(
        **{key: row[key] for key in ("shot", "query", "label", "disruptive")},
        before_cosine=float(np.clip(original @ r / (ns * nr), -1, 1))
        if ns and nr
        else None,
        after_cosine=float(np.clip(changed @ r / (nc * nr), -1, 1))
        if nc and nr
        else None,
        # Unlike a diagonal parameter-space preconditioner, weighting losses
        # may reverse descent for the original *equal-weight* objective.
        unweighted_signal_dot=float(original @ changed),
    )
