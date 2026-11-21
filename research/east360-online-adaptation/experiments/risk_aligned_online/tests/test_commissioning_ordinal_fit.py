import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_ordinal_20261111 as run


def test_ordered_difference_rejects_common_drift():
    disruption = torch.tensor([[.2, .8], [-.1, .4]])
    safe = torch.tensor([[.3, .5], [.0, .3]])
    expected = torch.tensor([.4, .2])
    torch.testing.assert_close(run.ordered_difference(disruption,safe),expected)
    both_shift = torch.tensor([.5, 1.1])
    torch.testing.assert_close(run.ordered_difference(disruption+both_shift,safe+both_shift),expected)
    torch.testing.assert_close(run.ordered_difference(disruption.flip(1),safe.flip(1)),-expected)


def test_random_order_control_balances_distinct_shots_and_keeps_each_shot_coherent():
    shots = np.array([1, 1, 1, 2, 2, 5, 6, 7, 7, 7, 7])
    s = run.shotwise_order_control(shots)
    np.testing.assert_array_equal(s,run.shotwise_order_control(shots))
    u = []
    for i in np.unique(shots):
        a = np.unique(s[shots==i])
        assert len(a)==1 and a[0] in (-1,1)
        u.append(a[0])
    assert abs(sum(u))<=1
