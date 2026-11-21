"""CPU-only linear optimization over a ball and nonnegative-cap halfspaces.

``solve`` returns a feasible numerical candidate and an upper bound for the
*linear* objective only.  Except for the feasible unconstrained-ball fast path,
optimality is not assumed: callers can inspect the returned primal/dual gap.
No statement is made about a nonlinear model or any experimental outcome.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import norm
from scipy.optimize import minimize, nnls


# A tiny absolute allowance prevents roundoff at a zero cap from collapsing an
# otherwise useful tangent direction to zero. It is below the 1e-7 contract.
_FEAS_TOL = 1e-11


def _ray_candidate(candidate, rows, caps, radius):
    """Scale toward the feasible origin, then check the original constraints.

    Positive caps and the ball are clipped along the ray. A positive residual at
    a zero cap is accepted only up to _FEAS_TOL; larger ones also clip along
    the ray. This is a radial feasibility repair, not a Euclidean projection.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    if not np.all(np.isfinite(candidate)):
        return None
    candidate_norm = float(norm(candidate))
    if not np.isfinite(candidate_norm):
        return None
    factor = min(1.0, radius / candidate_norm) if candidate_norm else 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = rows @ candidate
    if not np.all(np.isfinite(values)):
        return None
    clipped = (values > caps) & ((caps > 0.0) | (values > _FEAS_TOL))
    if np.any(clipped):
        factor = min(factor, float(np.min(caps[clipped] / values[clipped])))
    if factor < 1.0:
        candidate = candidate * np.nextafter(factor, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        values = rows @ candidate
    if (not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(values))
            or norm(candidate) > radius + _FEAS_TOL
            or np.any(values > caps + _FEAS_TOL)):
        return None
    return candidate


def _normalized_constraints(rows, caps, radius):
    """Return unit-normal constraints on v=u/radius, dropping redundancies."""
    scales = np.max(np.abs(rows), axis=1)
    keep = (scales > 0.0) & np.isfinite(caps)
    if not np.any(keep):
        return np.empty((0, rows.shape[1])), np.empty(0)
    scaled = rows[keep] / scales[keep, None]
    lengths = np.linalg.norm(scaled, axis=1)
    normals = scaled / lengths[:, None]
    # The wider intermediate avoids artificial overflow/underflow when the
    # radius, cap and row scale have very different magnitudes.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        bounds = np.asarray(
            caps[keep].astype(np.longdouble) / scales[keep].astype(np.longdouble)
            / lengths.astype(np.longdouble) / np.longdouble(radius),
            dtype=np.float64,
        )
    keep = bounds < 1.0  # Other rows follow from ||v|| <= 1.
    return normals[keep], bounds[keep]


def _upper_bound(g_unit, value_scale, normals, bounds, v):
    """Evaluate a nonnegative-multiplier linear dual; otherwise use the ball."""
    eps = np.finfo(np.float64).eps
    n, m = g_unit.size, bounds.size
    margin = 64.0 * eps * (1 + n + m)
    best = 1.0 + margin
    if m:
        # Inactive multipliers may be zero; NNLS only proposes a certificate.
        active = (bounds - normals @ v) <= 1e-6
        active_normals = normals[active]
        active_bounds = bounds[active]
        if active_bounds.size:
            columns = active_normals.T
            if norm(v) >= 1.0 - 1e-7:
                columns = np.column_stack((columns, v / norm(v)))
            try:
                coeff, _ = nnls(columns, g_unit, maxiter=10 * columns.shape[1])
                lam = np.maximum(coeff[:active_bounds.size], 0.0)
                remainder = g_unit - active_normals.T @ lam
                dual = float(norm(remainder) + active_bounds @ lam)
                # Pad for float64 dot/norm/normalization rounding. The bound is
                # a numerical certificate, not directed-rounding arithmetic.
                dual += margin * (1.0 + float(np.sum(lam)) + abs(dual))
                if np.isfinite(dual):
                    best = min(best, dual)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass
    upper = float(value_scale * best)
    if not np.isfinite(upper):
        raise ArithmeticError("linear upper bound is not finite")
    return float(np.nextafter(upper, np.inf))


def solve(g, rows, caps, radius):
    """Maximize ``g @ u`` subject to ``||u|| <= radius`` and ``rows @ u <= caps``.

    Parameters are finite real ``g`` (1-D), finite ``rows`` (m-by-n),
    nonnegative ``caps`` (length m; +inf denotes a disabled inequality), and a
    finite nonnegative radius. An empty row list is accepted. The returned pair
    is a float64 ndarray and Python float. The latter is a floating-point upper
    bound on the *linear* problem, using nonnegative dual multipliers when
    useful and the unconstrained-ball bound otherwise.

    The nontrivial path uses SLSQP. A near-zero residual on active zero-cap
    planes may first be removed by a least-squares correction; every candidate,
    including one returned after optimizer failure, is then independently
    repaired along its ray toward zero and checked against the original
    constraints, with an absolute 1e-11 numerical allowance. A failed optimizer
    emits RuntimeWarning if a useful checked direction is available, and
    otherwise raises RuntimeError. The primal/dual gap, not optimizer status
    alone, indicates solution quality.
    """
    g = np.asarray(g, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    caps = np.asarray(caps, dtype=np.float64)
    if g.ndim != 1:
        raise ValueError("g must be one-dimensional")
    if rows.ndim == 1 and rows.size == 0:
        rows = rows.reshape(0, g.size)
    if rows.ndim != 2 or rows.shape[1] != g.size:
        raise ValueError("rows must be a matrix with one column per element of g")
    if caps.ndim != 1 or caps.size != rows.shape[0]:
        raise ValueError("caps must have one element per row")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(rows)):
        raise ValueError("g and rows must be finite")
    if np.any(np.isnan(caps)) or np.any(caps < 0.0):
        raise ValueError("caps must be nonnegative and not NaN")
    try:
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("radius must be a finite nonnegative scalar") from exc
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be a finite nonnegative scalar")

    g_norm = float(norm(g)) if g.size else 0.0
    if not np.isfinite(g_norm):
        raise ArithmeticError("objective norm is not finite")
    if radius == 0.0 or g_norm == 0.0:
        return np.zeros_like(g), 0.0
    value_scale = g_norm * radius
    if not np.isfinite(value_scale):
        raise ArithmeticError("unconstrained linear upper bound is not finite")
    g_unit = g / g_norm
    ball = radius * g_unit
    with np.errstate(over="ignore", invalid="ignore"):
        ball_values = rows @ ball
    if (np.all(np.isfinite(ball_values)) and np.all(ball_values <= caps)
            and norm(ball) <= radius):
        return ball, _upper_bound(g_unit, value_scale, np.empty((0, g.size)), np.empty(0), g_unit)

    baseline = _ray_candidate(ball, rows, caps, radius)
    if baseline is None:
        raise ArithmeticError("could not construct a finite feasible ball direction")
    normals, bounds = _normalized_constraints(rows, caps, radius)
    if not bounds.size:
        return baseline, _upper_bound(g_unit, value_scale, normals, bounds, baseline / radius)

    constraints = (
        {"type": "ineq", "fun": lambda v: 1.0 - np.dot(v, v),
         "jac": lambda v: -2.0 * v},
        {"type": "ineq", "fun": lambda v: bounds - normals @ v,
         "jac": lambda v: -normals},
    )
    result = None
    failure = None
    try:
        result = minimize(
            lambda v: -float(g_unit @ v), baseline / radius,
            jac=lambda v: -g_unit, constraints=constraints, method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 200},
        )
        if not result.success:
            failure = str(result.message)
    except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        failure = str(exc)

    best = baseline
    best_value = float(g @ best)
    if result is not None:
        v = np.asarray(result.x, dtype=np.float64)
        if v.shape == g.shape and np.all(np.isfinite(v)):
            candidates = [v]
            # A zero cap can turn a microscopic, scaled-coordinate residual
            # into a large original-unit violation. Before radial repair,
            # remove roundoff from the active zero-cap planes together. This
            # also handles opposing rows that encode an equality.
            with np.errstate(over="ignore", invalid="ignore"):
                raw_values = rows @ (radius * v)
            active_zero = (bounds == 0.0) & (np.abs(normals @ v) <= 1e-8)
            if np.any((caps == 0.0) & (raw_values > _FEAS_TOL)) and np.any(active_zero):
                planes = normals[active_zero]
                try:
                    correction = np.linalg.lstsq(planes, planes @ v, rcond=None)[0]
                    candidates.append(v - correction)
                except np.linalg.LinAlgError:
                    pass
            checked = False
            for proposed in candidates:
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate = _ray_candidate(radius * proposed, rows, caps, radius)
                if candidate is not None:
                    candidate_value = float(g @ candidate)
                    if np.isfinite(candidate_value):
                        checked = True
                        if candidate_value > best_value:
                            best, best_value = candidate, candidate_value
            if not checked and failure is None:
                failure = "optimizer returned no finite checked candidate"
        elif failure is None:
            failure = "optimizer returned an invalid or nonfinite candidate"
    upper = _upper_bound(g_unit, value_scale, normals, bounds, best / radius)
    if not np.isfinite(best_value) or best_value < 0.0:
        raise ArithmeticError("no finite nonnegative feasible objective")
    if failure is not None:
        if best_value == 0.0 and upper > 0.0:
            raise RuntimeError(f"optimizer failed ({failure}); no useful feasible candidate")
        warnings.warn(f"optimizer failed ({failure}); returning checked feasible candidate", RuntimeWarning, stacklevel=2)
    return np.asarray(best, dtype=np.float64), upper
