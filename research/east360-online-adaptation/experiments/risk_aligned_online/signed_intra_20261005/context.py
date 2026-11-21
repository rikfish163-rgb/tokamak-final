"""Visible-only signed FFN action context (reports 17 section 6.3 and 20).

One shot per VisibleContext; call reset() before each new shot. This module
does not load data/checkpoints, train a rule, alter risk outputs, or schedule
events. The caller owns the existing P5+20 clock: origins 0, 26, ..., snapshots
at o+5/o+25, risk emitted BEFORE step(), earliest effect on the NEXT point.

Accept only copied, already-arrived [T, 23] raw signal/availability snapshots:
early h1..5 (T=5), late h5..25 (T=21). Late h5 is only a difference bridge,
never another level observation. Center/scale must be the frozen source
scaler, not statistics fitted from EAST. No labels, end time/length, evaluation
mask, or future waveform is accepted. Existing EAST evidence is dev-exposed.

Layout: [tanh(last z):23, tanh(mean adjacent dz):23, signed persistence:23,
         level-mask:23, difference-mask:23, current q:1, constant 1:1].
Rule rows follow ACTIONS = (0, -1, +1), also the exact argmax tie priority.
"""

import math

import torch


FEATURE_SIZE = 115
CONTEXT_SIZE = 117
ACTIONS = (0, -1, 1)


@torch.no_grad()
def segment_features(truth, observed, center, scale, segment):
    """Return detached [115] features in truth's dtype/device, without mutation.

    Valid means available AND finite, including genuine observed zeros. Use
    only original adjacent pairs, not differences after compressing out holes.
    Tanh follows aggregation for level/difference; persistence is the signed
    sum(dz)/(sum(abs(dz))+1e-12), with no extra tanh. Missing statistics are zero.
    Float64 intermediate arithmetic protects finite float32 extreme values.
    """
    if segment not in ("early", "late"):
        raise ValueError("declare early or late feedback segment")
    length = 5 if segment == "early" else 21
    if (truth.shape != (length, 23) or not truth.is_floating_point()
            or observed.shape != truth.shape or observed.dtype != torch.bool):
        raise ValueError("only arrived [T,23] floats and boolean mask: early h1..5, late h5..25")
    if center.shape != (23,) or scale.shape != (23,):
        raise ValueError("source center and scale must have shape [23]")
    if any(t.device != truth.device for t in (observed, center, scale)):
        raise ValueError("snapshot, availability and source scaler must share a device")
    if (not torch.isfinite(center).all() or not torch.isfinite(scale).all()
            or (scale <= 0).any()):
        raise ValueError("finite source center and positive finite scale required")

    valid = observed & torch.isfinite(truth)
    # Sanitize before arithmetic; NaN * 0 would still contaminate statistics.
    source_center, source_scale = center.double(), scale.double()
    values = torch.where(valid, truth.double(), source_center)
    z = torch.asinh((values - source_center) / source_scale)
    if not torch.isfinite(z).all():
        raise FloatingPointError("nonfinite source-normalized visible signal")
    level_valid = valid.clone()
    if segment == "late":
        level_valid[0] = False
    has_level = level_valid.any(0)
    indices = torch.arange(length, device=truth.device)[:, None]
    last_index = torch.where(level_valid, indices, -1).amax(0).clamp_min(0)
    last_level = z.gather(0, last_index[None])[0].masked_fill(~has_level, 0.)

    pairs = valid[1:] & valid[:-1]
    differences = (z[1:] - z[:-1]).masked_fill(~pairs, 0.)
    difference_sum = differences.sum(0)
    mean_difference = difference_sum / pairs.sum(0).clamp_min(1)
    persistence = difference_sum / (differences.abs().sum(0) + 1e-12)
    return torch.cat((last_level.tanh(), mean_difference.tanh(), persistence,
                      has_level.to(z.dtype), pairs.any(0).to(z.dtype))).to(truth.dtype)


def select_action(scores):
    """Exact argmax of three finite scores in row order (hold, negative, positive).

    The historical target's 1e-6 risk-loss tolerance is NOT an online logit
    tolerance; any strict score advantage wins here.
    """
    if scores.shape != (3,) or not torch.isfinite(scores).all():
        raise ValueError("three finite action scores required in order (0, -1, +1)")
    return ACTIONS[int(scores.argmax())]


def advance_q(q, action, segment):
    """One valid event: clip(q + eta * action, -1, 1), eta=.02/.08.

    Call only for nonempty scheduled events, never between them. A skipped
    event must not call this function (not even to project or validate q).
    """
    if type(action) is not int or action not in ACTIONS:
        raise ValueError("action must be the integer 0, -1 or +1")
    if segment not in ("early", "late"):
        raise ValueError("declare early or late feedback segment")
    if not math.isfinite(q) or not -1. <= q <= 1.:
        raise ValueError("finite q in [-1, 1] required")
    if action == 0:
        return q
    step = .02 if segment == "early" else .08
    return max(-1., min(1., q + step * action))


class VisibleContext:
    """Per-shot information ablation and scalar action state, not a risk model.

    H: live 69 signal features. P: freeze all 69 at the first valid event.
    L: live 23 levels, freeze 46 differences/persistence at that same event.
    All arms keep live 46 masks and their own current q, execute every valid
    event, and use the caller's same frozen [3,117] rule. Frozen signals are
    deliberately NOT re-zeroed by current masks; that would change report 20's
    ablation. Entirely missing events return (None, 0) without running the rule,
    initializing/replacing the freeze, or touching q. A bridge alone is empty.
    """

    def __init__(self, arm="H"):
        if arm not in ("H", "P", "L"):
            raise ValueError("information arm must be H, P or L")
        self.arm = arm
        self.reset()

    def reset(self):
        """Discard all temporary context and q at the shot boundary."""
        self.q = 0.
        self._first_signal = None

    @torch.no_grad()
    def step(self, truth, observed, center, scale, segment, rule):
        """Return (pre-action context [117], action), then expose updated self.q.

        Inputs are single-shot snapshots, not batches: squeeze the batch-of-one
        returned by partial_intra.stream.snapshot before calling. The returned
        context and cached first signal are independent copies. Rule may require
        gradients for OFFLINE training elsewhere; this online method detaches
        inputs and never mutates rule or the frozen source scaler.
        """
        features = segment_features(truth, observed, center, scale, segment)
        if not features[69:92].any():
            return None, 0
        if (rule.shape != (3, CONTEXT_SIZE) or not rule.is_floating_point()
                or not torch.isfinite(rule).all()):
            raise ValueError("finite floating-point rule [3,117] required")
        if rule.device != features.device or rule.dtype != features.dtype:
            raise ValueError("rule and snapshot must share dtype and device")

        first_signal = features[:69].clone() if self._first_signal is None else self._first_signal
        if self.arm == "P":
            features[:69] = first_signal
        elif self.arm == "L":
            features[23:69] = first_signal[23:69]
        context = torch.cat((features, features.new_tensor([self.q, 1.])))
        action = select_action(rule @ context)
        new_q = advance_q(self.q, action, segment)
        # Commit only after a valid rule/action; a failed event cannot seed P/L.
        self._first_signal = first_signal
        self.q = new_q
        return context, action
