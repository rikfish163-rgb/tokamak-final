import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.commissioning_lag_state_20261110 import LagState, replay
from experiments.risk_aligned_online.commissioning_transition_net_20261110 import move


def test_relation_contrast_fast_step_has_the_expected_gradient():
    torch.manual_seed(0)
    model = LagState(10, 3)
    a, k = model.design(torch.randn(2, 10))
    state = model.initial(2).requires_grad_()
    target = F.normalize(torch.randn(2, 3), dim=1)
    yes = torch.tensor([True, False])
    h = model.project(a, k, state)
    loss = F.softplus(-(model.decoder(h)*target).sum(-1))[0]
    exact = torch.autograd.grad(loss, state)[0]
    observed, _ = model.update(a, k, state, target, yes)
    torch.testing.assert_close(observed, move(state, exact, yes))


def test_replay_matches_prefix_and_null_has_same_release_and_count():
    torch.manual_seed(4)
    model = LagState(10, 4)
    rng = np.random.default_rng(131)
    x = rng.normal(size=(240, 10)).astype(np.float32)
    q = np.array([127, 159, 191])
    v = np.eye(4, dtype=np.float32)[:3]
    wrong = np.roll(v, 1, axis=1)
    m = np.array([True, False, True])
    out, audit = replay(model, x, q, v, wrong, m)
    short, _ = replay(model, x[:170], q[:1], v[:1], wrong[:1], m[:1])
    for name, rows in out.items():
        np.testing.assert_allclose(rows[:170], short[name], atol=1e-6)
        np.testing.assert_array_equal(rows[:153], out['fixed'][:153])
    assert audit['updates'] == {'real': 2, 'shifted': 2}
    assert not np.allclose(out['real'][153:], out['shifted'][153:])
