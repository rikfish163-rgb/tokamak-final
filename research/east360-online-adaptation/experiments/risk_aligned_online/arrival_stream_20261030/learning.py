"""Per-shot, signal-only causal streaming of temporary attention Q/K states.

This module never loads data or outcomes and never modifies the frozen model.
Each invocation starts a fresh shot. A receipt's ``apply_count`` counts points
actually scored while that maturity's replacement state was active, including
the next maturity point; it is zero when no subsequent point arrives.
"""

import numpy as np
import torch

from ..arrival_reference_20261030.run import ARMS, make_scores
from ..conditional_likelihood_20261029.neural import proposal
from ..conditional_likelihood_20261029.run import signal_windows
from ..delayed_recurrent_memory_20261025.learning import batch


_NAMES = ("cold",) + ARMS
_HORIZON = 25
_FIRST_ORIGIN = 127
_STATE_SIZE = 8192
_MAX_CHUNK = 32


def _prefix(signal, end):
    """Return a view containing only observations through inclusive ``end``."""
    return {name: value[:end + 1] for name, value in signal.items()}


def _anchor(net, signal, origin, slow):
    """Freeze the actual zero-Q/K origin CLS, risk logit, and Q/K gradient."""
    x, mask = batch(_prefix(signal, origin), [origin], net.base.center.device)
    captured = []
    handle = net.base.forecast_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    try:
        with torch.enable_grad():
            state = x.new_zeros((1, _STATE_SIZE), requires_grad=True)
            output = net(x, mask, state, slow, "qk", forecast=True)
            logit = output["logit"]
            if logit.shape != (1,) or not torch.isfinite(logit).all():
                raise FloatingPointError("invalid zero-Q/K origin logit")
            gradient, = torch.autograd.grad(logit.sum(), state)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (1, 17 * 64):
        raise ValueError("one 17-by-64 forecast-head input required at origin")
    feature = captured[0].reshape(1, 17, 64)[:, -1].detach()
    if not torch.isfinite(feature).all() or not torch.isfinite(gradient).all():
        raise FloatingPointError("nonfinite origin feature or Q/K gradient")
    return {"origin": origin, "feature": feature.cpu().numpy().copy(),
            "logit": logit.detach(), "gradient": gradient.detach()}


def _mature(net, signal, anchor, slow, pair, wrong_pair, reference, wrong_reference):
    """Use only the now-observed H25 to construct independent replacements."""
    origin = anchor["origin"]
    maturity = origin + _HORIZON
    observed = _prefix(signal, maturity)
    # signal_windows depends only on the origin and its H25 arrivals. Limiting
    # it to those 26 already-observed rows avoids rescanning the whole past at
    # every maturity; index zero is the original absolute ``origin``.
    horizon = {name: value[origin:] for name, value in observed.items()}
    target, valid = signal_windows(net, horizon, np.array([0], dtype=int))
    logs, counts, _ = make_scores(pair, wrong_pair, reference, wrong_reference,
                                  anchor["feature"], target, valid)
    if set(logs) != set(ARMS):
        raise ValueError("one feedback potential per declared arm required")
    potentials = np.concatenate([np.asarray(logs[name]) for name in ARMS], axis=0)
    counts = np.asarray(counts)
    if (potentials.shape != (len(ARMS), 2) or not np.isfinite(potentials).all()
            or counts.shape != (1,) or not np.isfinite(counts).all()
            or counts[0] < 0 or counts[0] > 23 or counts[0] != int(counts[0])):
        raise ValueError("finite per-arm potentials and supported-channel count required")
    device = anchor["gradient"].device
    potentials_t = torch.as_tensor(potentials, dtype=torch.float64, device=device)
    counts_t = torch.as_tensor(counts, device=device).expand(len(ARMS))
    replacements, feedback = proposal(anchor["logit"].expand(len(ARMS)),
        anchor["gradient"].expand(len(ARMS), -1), potentials_t, counts_t)
    zero = anchor["gradient"].new_zeros((1, _STATE_SIZE))
    states = torch.cat((zero, replacements), dim=0)
    norms = states.norm(dim=1).detach().cpu().numpy()
    feedback_values = feedback.detach().cpu().numpy()
    evidence = potentials[:, 1] - potentials[:, 0]
    if norms[_NAMES.index("identical")] != 0:
        raise ValueError("identical feedback must produce an exact zero Q/K state")
    receipt = {
        "origin": origin,
        "maturity": maturity,
        "first_affected": maturity + 1,
        "support_count": int(counts[0]),
        "state_norm": {name: float(value) for name, value in zip(_NAMES, norms)},
        "feedback": {"cold": 0.0, **{name: float(value)
                     for name, value in zip(ARMS, feedback_values)}},
        "e": {"cold": 0.0, **{name: float(value)
              for name, value in zip(ARMS, evidence)}},
        "origin_logit": float(anchor["logit"].item()),
        "gradnorm": float(anchor["gradient"].norm().item()),
        "apply_count": 0,
    }
    return states, receipt


def _forward(net, signal, start, stop, states, slow):
    """Score a bounded chunk via the same actual network path for all arms."""
    x, mask = batch(_prefix(signal, stop - 1), range(start, stop), net.base.center.device)
    count = len(x)
    # Duplicate only the last already-observed window to keep each arm's batch
    # aligned, even for a partial tail. Discard padded outputs. Fixed batch
    # geometry prevents row-position/tail-kernel rounding from making the
    # identical and cold paths differ or changing scores on a shorter prefix.
    if count < _MAX_CHUNK:
        x = torch.cat((x, x[-1:].expand(_MAX_CHUNK - count, -1, -1)))
        mask = torch.cat((mask, mask[-1:].expand(_MAX_CHUNK - count, -1, -1)))
    expanded = states[:, None, :].expand(-1, _MAX_CHUNK, -1).reshape(-1, _STATE_SIZE)
    with torch.no_grad():
        output = net(x.repeat(len(_NAMES), 1, 1), mask.repeat(len(_NAMES), 1, 1),
                     expanded, slow, "qk", forecast=False)
        logit = output["logit"]
        if logit.shape != (len(_NAMES) * _MAX_CHUNK,) or not torch.isfinite(logit).all():
            raise FloatingPointError("invalid actual neural risk logits")
        # Apply the identical operation to equally sized per-arm vectors.
        # Flattened CPU sigmoid can round its vectorized and scalar tail paths
        # differently when an arm boundary is not SIMD-aligned.
        rows = logit.reshape(len(_NAMES), _MAX_CHUNK)
        probabilities = torch.stack([row.sigmoid() for row in rows])
        if not torch.equal(probabilities[0], probabilities[_NAMES.index("identical")]):
            raise AssertionError("same-path identical and cold predictions differ")
        return probabilities[:, :count].cpu().numpy()


def predict(net, signal, slow, pair, wrong_pair, reference, wrong_reference):
    """Return every-point probabilities and compact receipts for one shot.

    ``signal`` must contain only ``values`` and boolean ``available`` arrays.
    Scores at each maturity are forwarded before the associated update. The
    origin anchor is always a separate zero-state forward, regardless of the
    state used to emit that point. No shot-level state is retained by the call.
    """
    if not isinstance(signal, dict) or set(signal) != {"values", "available"}:
        raise ValueError("signal-only values and available fields required")
    values, available = np.asarray(signal["values"]), np.asarray(signal["available"])
    if (values.ndim != 2 or values.shape[1] != 23 or not len(values)
            or values.dtype.kind not in "fiu" or available.shape != values.shape
            or available.dtype != np.bool_):
        raise ValueError("nonempty N-by-23 values and matching boolean availability required")
    signal = {"values": values, "available": available}
    states = net.base.center.new_zeros((len(_NAMES), _STATE_SIZE))
    scores = None
    receipts = []
    active_receipt = None
    anchor = None
    cursor, event = 0, _FIRST_ORIGIN
    while cursor < len(values):
        stop = min(cursor + _MAX_CHUNK, event + 1, len(values))
        probabilities = _forward(net, signal, cursor, stop, states, slow)
        if scores is None:
            scores = np.empty((len(_NAMES), len(values)), dtype=probabilities.dtype)
        scores[:, cursor:stop] = probabilities
        if active_receipt is not None:
            active_receipt["apply_count"] += stop - cursor
        cursor = stop
        if cursor == event + 1:
            if anchor is not None:
                states, active_receipt = _mature(net, signal, anchor, slow, pair,
                                                  wrong_pair, reference, wrong_reference)
                receipts.append(active_receipt)
            anchor = _anchor(net, signal, event, slow)
            event += _HORIZON
    return {name: scores[i] for i, name in enumerate(_NAMES)}, receipts
