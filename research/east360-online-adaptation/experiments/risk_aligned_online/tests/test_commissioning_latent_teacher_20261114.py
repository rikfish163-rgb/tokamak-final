import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_latent_teacher_20261114 as run
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_teacher_views_keep_the_same_prefix_and_masks_and_change_only_matured_lanes():
    x = torch.arange(2 * 128 * 23).reshape(2, 128, 23).float()
    m = torch.ones_like(x, dtype=torch.bool)
    m[:, :10, 5] = False
    m[:, -26, 5] = False
    m[:, -25:, 5] = True  # Not one of the eligible lanes.
    h, ok = run.teacher_views(x, m)
    torch.testing.assert_close(h[:, :, :-25], x[:, :-25].expand(3, -1, -1, -1))
    cols = list(run.embed.VALUES)
    rest = [k for k in range(23) if k not in cols]
    torch.testing.assert_close(h[:, :, -25:, rest],
                               x[:, -26:-25, rest].expand(3, -1, 25, -1))
    torch.testing.assert_close(h[0, :, -25:, cols], x[:, -25:, cols])
    torch.testing.assert_close(h[1, :, -25:, cols], x[:, -25:, cols].flip(1))
    torch.testing.assert_close(h[2, :, -25:, cols], x[:, -26:-25, cols].expand(-1, 25, -1))
    assert not ok[:, -25:, 5].any()
    torch.testing.assert_close(ok[:, -25:, cols], m[:, -25:, cols])


def test_a_constant_observed_block_does_not_create_a_latent_target():
    torch.manual_seed(6)
    base = Simple23(Config(), torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(3, 128, 23)
    x[:, -25:] = x[:, -26:-25]
    m = torch.ones_like(x, dtype=torch.bool)
    out = run.teacher_increment(base, x, m, lambda p: p)
    torch.testing.assert_close(out, torch.zeros(2, 3, 64), atol=3e-7, rtol=0.)


def test_ridge_sufficient_statistics_weight_short_and_long_shots_equally():
    r = np.random.default_rng(44)
    h = r.normal(size=(50, 3))
    g = r.normal(size=(50, 3))
    a = h @ r.normal(size=(3, 6)) + .15
    b = g @ r.normal(size=(3, 6)) - .35
    s = run.statistics(h, a)
    long = run.statistics(np.tile(h, (19, 1)), np.tile(a, (19, 1)))
    other = run.statistics(g, b)
    for i, j in zip(run.fit((s, other)), run.fit((long, other)), strict=True):
        np.testing.assert_allclose(i, j, atol=1e-12, rtol=0.)
    own = run.fit((s,))
    assert ((h @ own[0] + own[1] - a)**2).mean() < .05


def test_readout_and_orthogonal_objectives_are_complementary():
    v = torch.tensor([[1., 3., 4.], [1., 4., 5.]])
    beta = torch.tensor([1., 2., 0.])
    par = run.risk_component(v, beta)
    torch.testing.assert_close((v-par)@beta, torch.zeros(2))
    assert not torch.equal(par, v)


def test_selecting_earlier_orthogonal_gradient_does_not_change_its_objective():
    q = np.array([127, 159, 191, 223, 255])
    prev, eligible = run.embed.feedback_support(q, np.array([True, True, False, True, True]))
    g = torch.arange(7*6).reshape(7,6,1,1).float()
    base = run.assemble(g[:4],prev,run.MODES)
    extra = run.assemble(g,prev,(*run.MODES,*run.ORTHOGONAL_CONTROLS))
    torch.testing.assert_close(base,extra[:len(run.MODES)])
    assert eligible.tolist()==[False,True,False,False,True]
    torch.testing.assert_close(extra[5,2],g[3,1])
    torch.testing.assert_close(extra[6,2],g[4,2])
    torch.testing.assert_close(extra[7,2],g[5,2])
