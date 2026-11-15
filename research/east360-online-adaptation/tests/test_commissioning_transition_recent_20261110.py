import numpy as np
import torch

from experiments.risk_aligned_online.commissioning_transition_net_20261110 import TransitionNet, replay
from experiments.risk_aligned_online.commissioning_transition_recent_20261110 import recent_two


def test_recent_two_agrees_until_third_observation_and_is_prefix_causal():
    rng = np.random.default_rng(131)
    torch.manual_seed(131)
    net = TransitionNet(2)
    x = rng.normal(size=(263, 24)).astype(np.float32)
    q = np.array([127, 159, 191, 223])
    y = rng.integers(0, 3, size=(4, 5, 2))
    m = np.ones_like(y, dtype=bool)
    m[1] = False
    cumulative, _ = replay(net, x, q, y, m)
    rolling, observed = recent_two(net, x, q, y, m)
    truncated, _ = recent_two(net, x[:240], q[:3], y[:3], m[:3])
    assert observed == 3
    for k in rolling:
        np.testing.assert_allclose(rolling[k][:249], cumulative[k][:249], atol=1e-6)
        np.testing.assert_allclose(rolling[k][:240], truncated[k], atol=1e-6)
        assert not np.allclose(rolling[k][249:], cumulative[k][249:])
