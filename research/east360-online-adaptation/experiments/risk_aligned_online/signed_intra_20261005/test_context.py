"""CPU contracts for report 17 section 6.3 and report 20's information ablation."""

import numpy as np
import pytest
import torch

from ..partial_intra_20261005.feedback import event_clock
from ..partial_intra_20261005.stream import snapshot
from .context import (
    ACTIONS,
    CONTEXT_SIZE,
    FEATURE_SIZE,
    VisibleContext,
    advance_q,
    segment_features,
    select_action,
)


def block(segment="early"):
    length = 5 if segment == "early" else 21
    truth = torch.zeros(length, 23, dtype=torch.float64)
    return truth, torch.zeros_like(truth, dtype=torch.bool), torch.zeros(23), torch.ones(23)


def constant_rule(action):
    rule = torch.zeros(3, CONTEXT_SIZE, dtype=torch.float64)
    rule[ACTIONS.index(action), -1] = 1.
    return rule


def test_dimension_block_order_and_source_asinh_before_statistics():
    truth, mask, _, _ = block()
    center = torch.linspace(-3., 3., 23, dtype=truth.dtype)
    scale = torch.linspace(.5, 2., 23, dtype=truth.dtype)
    z = torch.arange(1., 24., dtype=truth.dtype)[None] / 100.
    z = z + torch.arange(5., dtype=truth.dtype)[:, None] / 10.
    truth = center + scale * z.sinh()
    mask[:] = True
    features = segment_features(truth, mask, center, scale, "early")
    expected = torch.cat((z[-1].tanh(), torch.full((23,), .1).double().tanh(),
                          torch.full((23,), .4 / (.4 + 1e-12), dtype=truth.dtype),
                          torch.ones(46, dtype=truth.dtype)))
    assert FEATURE_SIZE == 115 and CONTEXT_SIZE == 117
    assert features.shape == (115,) and features.dtype == truth.dtype
    torch.testing.assert_close(features, expected, rtol=1e-7, atol=1e-12)
    state = VisibleContext("H")
    context, action = state.step(truth, mask, center, scale, "early", constant_rule(0))
    torch.testing.assert_close(context[:115], features, rtol=0, atol=0)
    torch.testing.assert_close(context[115:], torch.tensor([0., 1.], dtype=truth.dtype))
    assert action == 0 and state.q == 0.


def test_last_actual_value_and_differences_do_not_cross_invalid_holes():
    truth, mask, center, scale = block()
    truth[:, 0] = torch.tensor([0., 1., 900., 4., 6.]).sinh()
    mask[:, 0] = torch.tensor([True, True, False, True, True])
    truth[:, 1] = torch.tensor([0., 2., 900., 900., 900.]).sinh()
    mask[:2, 1] = True
    truth[:, 2] = torch.tensor([1., 900., 3., 900., 5.]).sinh()
    mask[::2, 2] = True
    features = segment_features(truth, mask, center, scale, "early")
    assert features[0].item() == pytest.approx(np.tanh(6.))
    assert features[23].item() == pytest.approx(np.tanh(1.5))
    assert features[46].item() == pytest.approx(3. / (3. + 1e-12))
    assert features[1].item() == pytest.approx(np.tanh(2.))
    assert features[2].item() == pytest.approx(np.tanh(5.))
    assert features[25] == features[48] == features[94] == 0.
    assert features[69:72].tolist() == [1., 1., 1.]


@pytest.mark.parametrize("series, expected", [
    ([0., 1., 2., 3., 4.], 1.),
    ([4., 3., 2., 1., 0.], -1.),
    ([0., 1., 0., 1., 0.], 0.),
    ([2., 2., 2., 2., 2.], 0.),
])
def test_signed_persistence_is_not_absolute_or_tanh_transformed(series, expected):
    truth, mask, center, scale = block()
    truth[:, 0] = torch.tensor(series, dtype=truth.dtype).sinh()
    mask[:, 0] = True
    features = segment_features(truth, mask, center, scale, "early")
    assert features[46].item() == pytest.approx(expected, abs=1e-12)
    assert features[92] == 1.


def test_late_bridge_is_used_once_only_as_left_difference_endpoint():
    truth, mask, center, scale = block("late")
    truth[:2, 0] = torch.tensor([2., 3.]).sinh()
    mask[:2, 0] = True
    features = segment_features(truth, mask, center, scale, "late")
    assert features[0].item() == pytest.approx(np.tanh(3.))
    assert features[23].item() == pytest.approx(np.tanh(1.))
    assert features[92] == 1.
    mask[0, 0] = False
    no_bridge = segment_features(truth, mask, center, scale, "late")
    assert no_bridge[0] == features[0]
    assert no_bridge[23] == no_bridge[46] == no_bridge[92] == 0.
    mask[:] = False
    mask[0, 0] = True
    assert not segment_features(truth, mask, center, scale, "late").any()


def test_zero_is_observed_but_nonfinite_and_masked_values_are_missing():
    truth, mask, center, scale = block()
    mask[:] = True
    truth[:, 1] = float("nan")
    truth[:, 2] = float("inf")
    truth[:, 3] = -float("inf")
    truth[:, 4] = 999.
    mask[:, 4] = False
    features = segment_features(truth, mask, center, scale, "early")
    assert torch.isfinite(features).all()
    assert not features[:69].any()
    assert features[69] == features[92] == 1.
    assert not features[70:74].any() and not features[93:97].any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_features_are_detached_do_not_mutate_inputs_and_handle_extreme_float32(dtype):
    truth, mask, center, scale = block()
    truth = truth.to(dtype)
    truth[:, 0] = torch.tensor([-3e38, 3e38, -3e38, 3e38, 0.], dtype=dtype)
    truth.requires_grad_()
    mask[:, 0] = True
    before, before_mask = truth.detach().clone(), mask.clone()
    features = segment_features(truth, mask, center, scale, "early")
    assert torch.isfinite(features).all() and not features.requires_grad
    assert features.dtype == dtype and features.device.type == "cpu"
    assert torch.equal(truth, before) and torch.equal(mask, before_mask)


@pytest.mark.parametrize("segment, length", [("early", 4), ("early", 25), ("late", 20), ("late", 25)])
def test_only_exact_arrived_snapshot_shape_is_accepted(segment, length):
    truth = torch.zeros(length, 23)
    with pytest.raises(ValueError, match="arrived"):
        segment_features(truth, torch.ones_like(truth, dtype=torch.bool),
                         torch.zeros(23), torch.ones(23), segment)


@pytest.mark.parametrize("defect", ["segment", "channels", "mask_shape", "mask_dtype",
                                   "integer_truth", "center_shape", "center_nan", "scale_zero"])
def test_invalid_feature_contract_is_rejected(defect):
    truth, mask, center, scale = block()
    segment = "early"
    if defect == "segment": segment = "unknown"
    if defect == "channels": truth, mask = truth[:, :22], mask[:, :22]
    if defect == "mask_shape": mask = mask[:1]
    if defect == "mask_dtype": mask = mask.float()
    if defect == "integer_truth": truth = truth.long()
    if defect == "center_shape": center = center[:22]
    if defect == "center_nan": center[0] = float("nan")
    if defect == "scale_zero": scale[0] = 0.
    with pytest.raises(ValueError):
        segment_features(truth, mask, center, scale, segment)


@pytest.mark.parametrize("arm", ["H", "P", "L"])
def test_ablation_freezes_only_specified_signals_and_keeps_live_masks_and_q(arm):
    state = VisibleContext(arm)
    truth, mask, center, scale = block()
    truth[:, 0] = torch.arange(5., dtype=truth.dtype).sinh()
    mask[:, 0] = True
    first, _ = state.step(truth, mask, center, scale, "early", constant_rule(1))
    first_saved = first.clone()
    truth[:].fill_(900.)  # Snapshots and returned contexts must not alias the freeze.
    first[:69] = -100.
    late, late_mask, _, _ = block("late")
    late[-1, 1] = np.sinh(-2.)
    late_mask[-1, 1] = True
    second, _ = state.step(late, late_mask, center, scale, "late", constant_rule(1))
    live = segment_features(late, late_mask, center, scale, "late")
    torch.testing.assert_close(second[69:115], live[69:], rtol=0, atol=0)
    if arm == "H":
        torch.testing.assert_close(second[:69], live[:69], rtol=0, atol=0)
    elif arm == "P":
        torch.testing.assert_close(second[:69], first_saved[:69], rtol=0, atol=0)
    else:
        torch.testing.assert_close(second[:23], live[:23], rtol=0, atol=0)
        torch.testing.assert_close(second[23:69], first_saved[23:69], rtol=0, atol=0)
    assert second[115].item() == .02 and second[116] == 1.
    assert state.q == pytest.approx(.10)


@pytest.mark.parametrize("arm", ["P", "L"])
def test_frozen_information_still_runs_rule_with_its_own_current_q(arm):
    truth, mask, center, scale = block()
    mask[:] = True
    state = VisibleContext(arm)
    rule = constant_rule(1)
    rule[ACTIONS.index(1), 115] = -100.
    rule[ACTIONS.index(-1), 115] = 100.
    first, a1 = state.step(truth, mask, center, scale, "early", rule)
    second, a2 = state.step(truth, mask, center, scale, "early", rule)
    assert first[115] == 0. and second[115].item() == .02
    assert (a1, a2) == (1, -1) and state.q == 0.


@pytest.mark.parametrize("arm", ["H", "P", "L"])
@pytest.mark.parametrize("segment", ["early", "late"])
def test_empty_events_do_not_initialize_freeze_run_rule_or_update_q(arm, segment):
    state = VisibleContext(arm)
    empty, mask, center, scale = block(segment)
    empty[:] = float("nan")
    mask[:] = True
    if segment == "late": empty[0] = 5.  # A bridge alone is not a new observation.
    invalid_rule = torch.full((3, CONTEXT_SIZE), float("nan"), dtype=empty.dtype)
    assert state.step(empty, mask, center, scale, segment, invalid_rule) == (None, 0)
    assert state.q == 0.
    truth, observed, _, _ = block()
    truth[:, 0] = np.sinh(2.)
    observed[:, 0] = True
    first, _ = state.step(truth, observed, center, scale, "early", constant_rule(1))
    assert first[0].item() == pytest.approx(np.tanh(2.))
    q_before = state.q
    assert state.step(empty, mask, center, scale, segment, invalid_rule) == (None, 0)
    assert state.q == q_before
    next_context, _ = state.step(truth, observed, center, scale, "early", constant_rule(0))
    torch.testing.assert_close(next_context[:115], first[:115], rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["H", "P", "L"])
def test_reset_clears_q_and_first_event_between_shots(arm):
    truth, mask, center, scale = block()
    mask[:] = True
    state = VisibleContext(arm)
    state.step(truth, mask, center, scale, "early", constant_rule(1))
    state.reset()
    assert state.q == 0.
    truth[:] = np.sinh(-2.)
    actual, action = state.step(truth, mask, center, scale, "early", constant_rule(0))
    expected, _ = VisibleContext(arm).step(truth, mask, center, scale, "early", constant_rule(0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert action == 0


def prefix_trace(values, available, cutoff, arm):
    state = VisibleContext(arm)
    rows = []
    signals = dict(values=values, available=available)
    for origin in range(0, cutoff + 1, 26):
        for arrived, update, effective, segment in event_clock(origin, "partial"):
            if arrived > cutoff:
                continue
            truth, observed = snapshot(signals, origin, segment, "cpu", torch.float64)
            context, action = state.step(truth[0], observed[0], torch.zeros(23),
                                         torch.ones(23), segment, constant_rule(1))
            assert update == arrived and effective == update + 1
            rows.append((arrived, context, action, state.q))
    return rows


@pytest.mark.parametrize("arm", ["H", "P", "L"])
@pytest.mark.parametrize("cutoff", [4, 5, 24, 25, 26, 31, 51])
def test_future_values_masks_and_final_length_cannot_change_arrived_context(arm, cutoff):
    rng = np.random.default_rng(131)
    values = rng.normal(size=(80, 23))
    available = rng.random((80, 23)) > .3
    original = prefix_trace(values, available, cutoff, arm)
    mutated_values, mutated_mask = values.copy(), available.copy()
    mutated_values[cutoff + 1:] = float("nan")
    mutated_mask[cutoff + 1:] = ~mutated_mask[cutoff + 1:]
    variants = [prefix_trace(mutated_values, mutated_mask, cutoff, arm),
                prefix_trace(values[:cutoff + 1], available[:cutoff + 1], cutoff, arm)]
    for result in variants:
        assert len(result) == len(original)
        for before, after in zip(original, result):
            assert before[0] == after[0] and before[2:] == after[2:]
            torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)


@pytest.mark.parametrize("scores, expected", [
    ([0., 0., 0.], 0), ([1., 1., 0.], 0), ([1., 0., 1.], 0),
    ([0., 1., 1.], -1), ([0., 0., 1.], 1), ([0., 1e-12, 0.], -1),
])
def test_online_action_order_and_exact_tie_breaking(scores, expected):
    assert ACTIONS == (0, -1, 1)
    assert select_action(torch.tensor(scores, dtype=torch.float64)) == expected


@pytest.mark.parametrize("segment, step", [("early", .02), ("late", .08)])
@pytest.mark.parametrize("action", [-1, 0, 1])
def test_q_uses_fixed_segment_steps_clips_and_does_not_decay(segment, step, action):
    assert advance_q(0., action, segment) == pytest.approx(action * step)
    for q in (-1., -.99, -.2, 0., .2, .99, 1.):
        updated = advance_q(q, action, segment)
        assert -1. <= updated <= 1. and abs(updated - q) <= step + 1e-15
        if action == 0: assert updated == q
    q = 0.
    for _ in range(60): q = advance_q(q, action, segment)
    assert q == float(action)


@pytest.mark.parametrize("action", [True, 2, .5])
def test_invalid_actions_are_rejected(action):
    with pytest.raises(ValueError): advance_q(0., action, "early")


@pytest.mark.parametrize("q", [float("nan"), float("inf"), -1.01, 1.01])
def test_invalid_q_is_rejected(q):
    with pytest.raises(ValueError): advance_q(q, 0, "early")


def test_invalid_arm_rule_logits_and_segment_are_rejected():
    with pytest.raises(ValueError): VisibleContext("hold")
    with pytest.raises(ValueError): advance_q(0., 1, "unknown")
    for scores in (torch.zeros(2), torch.tensor([0., float("nan"), 1.])):
        with pytest.raises(ValueError): select_action(scores)
    truth, mask, center, scale = block()
    mask[:] = True
    for rule in (torch.zeros(2, 117), torch.full((3, 117), float("nan"))):
        with pytest.raises(ValueError):
            VisibleContext().step(truth, mask, center, scale, "early", rule)
