import numpy as np
import pytest
import torch
import json
from types import SimpleNamespace

from .finite_probe import fixed_sign_curve, bias_fit, choose_historical_action, predict_fixed, whole_cost, reference_order
from ..simple_online import _hash
from ..redesign_20261004.prepare_prior import PARENT_SHA256


def test_fixed_curve_emits_before_update_and_has_both_partial_steps():
    q = fixed_sign_curve(np.ones((55, 23), bool), 1)
    np.testing.assert_array_equal(q[:6], 0.)
    np.testing.assert_allclose(q[6:26], .02)
    np.testing.assert_allclose(q[26:32], .1)
    np.testing.assert_allclose(q[32:52], .12)
    np.testing.assert_allclose(q[52:], .2)


def test_no_future_mask_dependency():
    mask = np.ones((80, 23), bool)
    q = fixed_sign_curve(mask, -1)
    for n in (4, 6, 25, 26, 40, 55):
        np.testing.assert_array_equal(q[:n], fixed_sign_curve(mask[:n], -1))


def test_empty_segment_no_update_and_bridge_is_not_reused_level():
    mask = np.zeros((28, 23), bool); mask[5, 0] = True
    q = fixed_sign_curve(mask, 1)
    assert q[6] == pytest.approx(.02)
    assert q[26] == pytest.approx(.02)
    assert not fixed_sign_curve(np.zeros((40, 23), bool), 1).any()


def test_bounded_cumulative_state_and_hold_identity():
    a = np.ones((600, 23), bool)
    assert max(fixed_sign_curve(a, 1)) == 1.
    assert min(fixed_sign_curve(a, -1)) == -1.
    assert not fixed_sign_curve(a, 0).any()


@pytest.mark.parametrize('sign', [2, .5, True])
def test_bad_action_rejected(sign):
    with pytest.raises(ValueError): fixed_sign_curve(np.ones((10, 23), bool), sign)


def test_historical_action_uses_error_before_surrogate_and_hold_ties():
    assert choose_historical_action({-1: (0, .2), 0: (1, .1), 1: (1, .01)}) == -1
    assert choose_historical_action({-1: (0, .2), 0: (0, .2), 1: (0, .2)}) == 0
    assert choose_historical_action({-1: (0, .1999999), 0: (0, .2), 1: (0, .3)}) == 0


def test_bias_fit_exact_and_empty_and_equal_shot_weights():
    assert bias_fit([(np.ones(3), np.zeros(3))])['energy_fraction'] is None
    a = bias_fit([(np.array([.1, .2]), np.array([.4, .8])),
                  (np.array([.2]), np.array([.8]))])
    assert a['beta'] == pytest.approx(4.)
    assert a['energy_fraction'] == pytest.approx(1.)
    # Equal weight per shot, not per endpoint: beta=(1+2)/2=1.5.
    b = bias_fit([(np.ones(100), np.ones(100)), (np.ones(1), np.ones(1)*2)])
    assert b['beta'] == pytest.approx(1.5)


class ToyModel:
    center = torch.zeros(23)
    cfg = SimpleNamespace(window=128)

    def __call__(self, values, available, delta):
        return {'logit': values[:, -1, 0] + delta.sum()}


def test_predict_fixed_causal_prefix_reset_and_absent_outcomes():
    shot = dict(device='EAST', values=np.zeros((65, 23), np.float32),
                available=np.ones((65, 23), bool), times=np.arange(65)*.001)
    direction = torch.tensor([1., 0.])
    full = predict_fixed(ToyModel(), shot, direction, 1)
    shorter = {k: v[:40].copy() if isinstance(v, np.ndarray) else v for k, v in shot.items()}
    prefix = predict_fixed(ToyModel(), shorter, direction, 1)
    for a, b in zip(full, prefix): np.testing.assert_array_equal(a[:40], b)
    shot['values'][40:] = 900.
    changed = predict_fixed(ToyModel(), shot, direction, 1)
    for a, b in zip(full, changed): np.testing.assert_array_equal(a[:40], b[:40])
    # Same model object, fresh call, no inherited temporary state.
    again = predict_fixed(ToyModel(), shorter, direction, 1)
    for a, b in zip(prefix, again): np.testing.assert_array_equal(a, b)
    assert full[0][5] == 0 and full[0][6] == pytest.approx(.02)


def test_whole_cost_keeps_late_miss_and_strict30():
    shot = dict(shot=1, is_disrupt=True, down=1., times=np.array([.96, .97, 1.01]),
                valid=np.ones(3, bool))
    row, cost, prediction = whole_cost(shot, np.array([.1, .9, .99]), .5)
    assert cost[0] == 1 and not prediction['timely']
    assert len(row['times']) == 2 and prediction['lead_ms'] == 30.


def test_no_timely_endpoint_does_not_remove_disruptive_shot():
    shot = dict(shot=1, is_disrupt=True, down=1., times=np.array([.99]), valid=np.ones(1, bool))
    row, cost, prediction = whole_cost(shot, np.array([.1]), .5)
    assert cost == (1, None) and row['is_disrupt']


def test_reference_locks_original_manifest_and_checks_copied_semantics(tmp_path):
    root = tmp_path/'run'; root.mkdir(); static = root/'static'; static.mkdir()
    original = tmp_path/'order.json'
    rows = [{'shot': i+1, 'order_method': 'shot_number'} for i in range(2953)]
    original.write_text(json.dumps(rows, separators=(',', ':')))
    (root/'order.json').write_text(json.dumps(rows, indent=2))
    protocol = dict(source_sha256=PARENT_SHA256, rate_contract='nominal',
                    order_sha256=_hash(original), args=dict(order=str(original)))
    (root/'protocol.json').write_text(json.dumps(protocol))
    assert reference_order(static) == rows
    rows[0]['shot'] = 999999
    (root/'order.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError): reference_order(static)
