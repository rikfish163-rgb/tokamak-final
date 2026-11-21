import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.signed_intra_20261005.credit_span import single_event_plan
from experiments.risk_aligned_online.signed_intra_20261005.finite_probe import fixed_sign_curve


@pytest.mark.parametrize('sign', [-1, 1])
def test_single_action_is_persistent_not_reset_or_repeated(sign):
    signals = dict(device='EAST', values=np.ones((300, 23)), available=np.ones((300, 23), bool))
    center, scale = torch.zeros(23), torch.ones(23)
    q, events = single_event_plan(signals, center, scale, sign)
    assert np.all(q[:6] == 0)
    assert np.all(q[6:] == sign*.02)
    assert events[0]['effect'] == 6
    assert events[0]['action'] == sign
    assert all(e['action'] == 0 for e in events[1:])
    assert fixed_sign_curve(signals['available'], sign)[-1] == sign
    for length in (7, 27, 70):
        prefix = {k: v[:length] if k in ('values', 'available') else v for k, v in signals.items()}
        truncated, _ = single_event_plan(prefix, center, scale, sign)
        assert np.array_equal(truncated, q[:length])


def test_single_plan_requires_signed_existing_event():
    signals = dict(device='EAST', values=np.zeros((30, 23)), available=np.zeros((30, 23), bool))
    with pytest.raises(ValueError):
        single_event_plan(signals, torch.zeros(23), torch.ones(23), 1)
    with pytest.raises(ValueError):
        single_event_plan(signals, torch.zeros(23), torch.ones(23), 0)
