import copy

import pytest
import torch

from experiments.risk_aligned_online.maturity_jacobian_20261030.run import (
    mature_data,
    promotion,
)


def test_only_update_window_changes_and_cannot_touch_future():
    full = torch.arange(178 * 23, dtype=torch.float32).reshape(1, 178, 23)
    data = dict(x=full[:, :128].clone(), mask=torch.ones(1, 128, 23, dtype=torch.bool),
                later_x=full[:, 50:].clone(), later_mask=torch.ones(1, 128, 23, dtype=torch.bool),
                context=torch.ones(1, 358), support=torch.ones(1, dtype=torch.bool))
    records = [dict(origin=127, maturity=152, query=177)]
    actual = mature_data(data, records)
    assert torch.equal(actual['x'], full[:, 25:153])
    assert actual['context'] is data['context']
    assert actual['later_x'] is data['later_x']
    data['later_x'][:, -25:] = float('nan')
    assert torch.equal(mature_data(data, records)['x'], actual['x'])
    assert torch.equal(data['x'], full[:, :128])


def test_wrong_clock_is_rejected():
    with pytest.raises(ValueError, match='clock'):
        mature_data({}, [dict(origin=127, maturity=151, query=177)])


def good():
    groups = {arm: {group: dict(mean_delta_bce=value, improved=20)
                   for group in ('safe', 'danger')}
              for arm, value in (('real', -.2), ('reverse', -.1), ('donor', -.1))}
    old = copy.deepcopy(groups)
    old['real']['safe']['mean_delta_bce'] = -.15
    old['real']['danger']['mean_delta_bce'] = -.15
    return groups, old


def test_promotion_needs_both_old_comparisons_and_specificity():
    groups, old = good()
    assert promotion(groups, old, .8, .9)['passed']
    old['real']['safe']['mean_delta_bce'] = -.3
    result = promotion(groups, old, .8, .9)
    assert not result['passed']
    assert 'safe: not better than origin-teacher/origin-step' in result['reasons']
    assert not promotion(groups, old, .999, .9)['passed']


def test_equality_to_old_is_not_success():
    groups, old = good()
    old['real']['danger']['mean_delta_bce'] = -.2
    assert not promotion(groups, old, .8, .9)['passed']


def test_actual_inner_gradient_uses_selected_window_not_query_future():
    from experiments.risk_aligned_online.anchored_feedback_20261030.run import state_for

    class Net:
        def first(self, x, mask, state, site):
            assert site == 'qk'
            coefficients = x.mean(1).repeat(1, 3)[:, :64]
            return (coefficients * state[:, :64])[:, None], None, None

    class Target:
        def __call__(self, context, support, mean):
            return torch.ones(len(context), 64)

    full = torch.arange(178 * 23, dtype=torch.float32).reshape(1, 178, 23)
    data = dict(x=full[:, :128], mask=torch.ones(1, 128, 23, dtype=torch.bool),
                later_x=full[:, 50:].clone(), later_mask=torch.ones(1, 128, 23, dtype=torch.bool),
                context=torch.zeros(1, 358), support=torch.ones(1, dtype=torch.bool))
    records = [dict(origin=127, maturity=152, query=177)]
    mature = mature_data(data, records)
    args = (Net(), Target(), 'anchored')
    old = state_for(*args, data, data['context'], torch.zeros(358))
    new = state_for(*args, mature, data['context'], torch.zeros(358))
    assert not torch.allclose(old, new, atol=1e-8, rtol=1e-6)
    expected = mature['x'].mean(1).repeat(1, 3)[:, :64]
    expected = .1 * expected / expected.double().norm(dim=1, keepdim=True).float()
    torch.testing.assert_close(new[:, :64], expected)
    assert torch.count_nonzero(new[:, 64:]) == 0
    data['later_x'][:, -25:] = float('nan')
    changed = mature_data(data, records)
    again = state_for(*args, changed, data['context'], torch.zeros(358))
    assert torch.equal(new, again)
