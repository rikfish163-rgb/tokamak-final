"""CPU-only consistency checks for bounded Armijo receipts, with no file I/O.

This module imports neither the neural model nor the solver. It checks recorded
losses/decisions, not their truth: no neural loss or actual hidden RMS is rerun.
The raw metric direction is not recorded, so rejected steps' directional
derivatives cannot be independently recovered. Accepted steps admit an extra
gradient/retained-state bound check, allowing for floating-point state rounding.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

RTOL, ATOL, ARMIJO_C = 1e-9, 1e-14, 1e-4


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _scalar(value, name):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)), name + ": finite number required")
    result = float(value)
    _require(np.isfinite(result), name + ": nonfinite")
    return result


def _integer(value, maximum, name):
    _require(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and 0 <= value <= maximum,
        name + ": invalid count",
    )
    return int(value)


def _vector(value, name):
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + ": numeric flat256 required") from error
    _require(result.shape == (256,) and result.dtype.kind in "fiu", name + ": numeric flat256 required")
    if result.dtype.kind in "iu":
        result = result.astype(np.float64)
    _require(result.dtype.itemsize in (4, 8) and np.isfinite(result).all(), name + ": finite float32/64 required")
    return result


def _same_record(declared, actual, name):
    declared = _vector(declared, name)
    _require(
        np.array_equal(declared.astype(np.float64), actual.astype(np.float64))
        and np.array_equal(np.signbit(declared), np.signbit(actual)),
        name + ": recorded vector mismatch",
    )


def _close(left, right, name, *, extra_atol=0.0):
    _require(np.isclose(left, right, rtol=RTOL, atol=ATOL + extra_atol), name + ": numeric mismatch")


def _summary(receipt, observed, gradient_norm, step_norm, placeholder, rejected):
    return dict(
        events=1,
        observed=observed,
        observed_scalars=observed,
        evaluations=int(receipt["evaluations"]),
        accepted=receipt["accepted"],
        reason=receipt["reason"],
        alpha=float(receipt["alpha"]),
        gradient_computed=observed > 0,
        gradient_placeholder=placeholder,
        nonzero_gradients=int(observed > 0 and gradient_norm > 0),
        gradient_norm=gradient_norm,
        step_norm=step_norm,
        rejected_trials=rejected,
        neural_loss_recomputed=False,
        hidden_rms_verified=False,
    )


def verify_receipt(before256, after256, gradient256, receipt, observed):
    """Validate one propose/export receipt; return a fixed aggregation summary.

    Arrays must be finite flat256 vectors. Before/after must retain the same
    float32/64 storage dtype, since the solver subtracts states in that dtype
    before computing a double-precision norm. Norm tolerance is rtol=1e-9,
    atol=1e-14, matching full_path.verify. Rejections must leave state bitwise
    unchanged. Acceptance inequalities themselves are strict/exact, not relaxed
    by that tolerance. Every trial must follow alpha=2**(-i), at most 11 trials.

    ``receipt`` contains select_step's fields plus propose's gradient_norm and
    step_norm. Exported receipts legitimately omit gradient256/state256_after;
    both are optional, but checked exactly against the supplied vectors when
    present. Additional trace metadata is allowed. A no_target gradient may be
    None or an all-zero storage sentinel, never a measured zero gradient.
    ``observed`` is an integer in [0,575]; mask/causal-clock verification belongs
    to the caller, including the deployed stream's tighter 550-scalar support.

    Fixed summary keys:
      events, observed, observed_scalars, evaluations, accepted, reason, alpha,
      gradient_computed, gradient_placeholder, nonzero_gradients, gradient_norm,
      step_norm, rejected_trials, neural_loss_recomputed, hidden_rms_verified.
    accepted is bool; events/nonzero_gradients/counts are integers. The final
    two verification flags are always False. Inputs are never mutated.
    """
    _require(isinstance(receipt, Mapping), "receipt mapping required")
    _require(
        {"accepted", "reason", "alpha", "evaluations", "trials", "gradient_norm", "step_norm"} <= receipt.keys(),
        "missing receipt fields",
    )
    _require("loss_proposal" not in receipt, "legacy loss_proposal is not an Armijo receipt field")
    observed = _integer(observed, 575, "observed")
    evaluations = _integer(receipt["evaluations"], 11, "evaluations")
    _require(type(receipt["accepted"]) is bool and isinstance(receipt["reason"], str), "accepted/reason types")
    _require(isinstance(receipt["trials"], list) and len(receipt["trials"]) == evaluations, "trial count")
    alpha = _scalar(receipt["alpha"], "alpha")
    before, after = _vector(before256, "before256"), _vector(after256, "after256")
    _require(before.dtype.str == after.dtype.str, "state storage dtype mismatch")
    unchanged = before.tobytes() == after.tobytes()
    with np.errstate(over="ignore", invalid="ignore"):
        delta = after - before
        step_norm = float(np.linalg.norm(delta.astype(np.float64)))
    _require(np.isfinite(delta).all() and np.isfinite(step_norm), "nonfinite state difference/norm")
    gradient = None if gradient256 is None else _vector(gradient256, "gradient256")
    with np.errstate(over="ignore", invalid="ignore"):
        gradient_norm = 0.0 if gradient is None else float(np.linalg.norm(gradient.astype(np.float64)))
    _require(np.isfinite(gradient_norm), "nonfinite gradient norm")
    recorded_g = _scalar(receipt["gradient_norm"], "gradient_norm")
    recorded_step = _scalar(receipt["step_norm"], "step_norm")
    _require(recorded_g >= 0 and recorded_step >= 0, "negative norm")
    _close(recorded_g, gradient_norm, "gradient_norm")
    _close(recorded_step, step_norm, "step_norm")
    if "state256_after" in receipt:
        _same_record(receipt["state256_after"], after, "state256_after")
    if "observed" in receipt:
        _require(_integer(receipt["observed"], 575, "receipt observed") == observed, "observed mismatch")
    if "gradient_computed" in receipt:
        _require(
            type(receipt["gradient_computed"]) is bool and receipt["gradient_computed"] == (observed > 0),
            "gradient measurement flag",
        )

    if not observed:
        _require(
            receipt["reason"] == "no_target"
            and not receipt["accepted"]
            and unchanged
            and evaluations == 0
            and alpha == recorded_g == recorded_step == 0
            and (gradient is None or not np.any(gradient)),
            "invalid no_target state/placeholder",
        )
        _require(
            not {"loss_before", "loss_after", "armijo_bound", "gradient256"}.intersection(receipt),
            "no_target must not claim a measured loss or gradient",
        )
        return _summary(receipt, observed, 0.0, 0.0, gradient is not None, 0)

    _require(gradient is not None, "observed target requires a measured gradient vector")
    if "gradient256" in receipt:
        _same_record(receipt["gradient256"], gradient, "receipt gradient256")
    _require({"loss_before", "loss_after", "armijo_bound"} <= receipt.keys(), "missing observed loss fields")
    old, retained = _scalar(receipt["loss_before"], "loss_before"), _scalar(receipt["loss_after"], "loss_after")
    zero_gradient = not np.any(gradient)
    trials, passing, bounds = receipt["trials"], [], []
    for index, trial in enumerate(trials):
        _require(isinstance(trial, Mapping) and {"alpha", "loss", "armijo_bound"} <= trial.keys(), "trial fields")
        trial_alpha = _scalar(trial["alpha"], "trial alpha")
        loss, bound = _scalar(trial["loss"], "trial loss"), _scalar(trial["armijo_bound"], "trial bound")
        _require(trial_alpha == 2.0 ** (-index) and bound <= old, "trial alpha/descent bound")
        if bounds:
            _close(bound, old + trial_alpha * (bounds[0] - old), "bound halving schedule")
            if bounds[0] == old:
                _require(bound == old, "rounded-zero bound schedule")
        bounds.append(bound)
        if loss < old and loss <= bound:
            passing.append(index)

    if receipt["accepted"]:
        _require(
            receipt["reason"] == "armijo"
            and evaluations > 0
            and not zero_gradient
            and not unchanged
            and passing == [evaluations - 1],
            "must accept the first passing trial and stop",
        )
        last = trials[-1]
        selected_bound = _scalar(receipt["armijo_bound"], "selected bound")
        _require(
            alpha == last["alpha"] and retained == last["loss"] and selected_bound == last["armijo_bound"],
            "selected trial fields mismatch",
        )
        # Recover the raw direction only up to alpha*d addition/subtraction
        # rounding; do not confuse a stored float32 W increment with exact d.
        with np.errstate(over="ignore", invalid="ignore"):
            ulps = sum(np.abs(x - np.nextafter(x, np.zeros_like(x))).astype(np.float64) for x in (before, after, delta))
            dot = float(np.dot(gradient.astype(np.float64), delta.astype(np.float64)))
            rounding = float(np.dot(np.abs(gradient.astype(np.float64)), ulps))
            expected = old + ARMIJO_C * dot / alpha
        _require(np.isfinite(expected) and np.isfinite(rounding), "nonfinite reconstructed step bound")
        _close(bounds[0], expected, "gradient/state directional bound", extra_atol=ARMIJO_C * rounding / alpha)
        return _summary(receipt, observed, gradient_norm, step_norm, False, evaluations - 1)

    _require(
        unchanged
        and step_norm == recorded_step == 0
        and alpha == 0
        and retained == old
        and receipt["armijo_bound"] is None
        and not passing,
        "invalid rejected receipt",
    )
    reason = receipt["reason"]
    if reason == "stationary":
        _require(zero_gradient and evaluations == 0, "stationary requires a measured zero gradient")
    elif reason == "no_effective_direction":
        _require(not zero_gradient and evaluations == 0, "no_effective_direction requires nonzero gradient, no trials")
    elif reason == "budget_exhausted":
        _require(not zero_gradient and evaluations == 11, "budget_exhausted requires all 11 failed trials")
    else:
        raise ValueError("unknown observed rejection reason: " + reason)
    return _summary(receipt, observed, gradient_norm, step_norm, False, evaluations)
