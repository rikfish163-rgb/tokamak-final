import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from trajectory_policy_20261105 import (
    chronological_queries,
    contrast_features,
    effect_gradient,
    routed_pair,
    window_views,
)


def test_windows_do_not_see_future_samples_and_use_exact_causal_shape():
    values = torch.arange(170.0)[:, None].repeat(1, 23)
    mask = torch.ones_like(values, dtype=torch.bool)
    x, a = window_views(values, mask)
    assert x.shape == a.shape == (170, 128, 23)
    assert x[127, 0, 0] == 0 and x[127, -1, 0] == 127
    assert not a[126, 0].any()
    old = x[135].clone()
    values[136:] = -900
    xnew, _ = window_views(values, mask)
    assert torch.equal(xnew[135], old)


def test_actual_vs_lagged_features_distinguish_recent_only_motion():
    v = torch.zeros(1, 128, 23)
    a = torch.ones_like(v, dtype=torch.bool)
    v[:, -16:, 0] = 2
    result = contrast_features(v, a, torch.tensor([0.2]))
    assert set(result) == {"real", "lagged", "state"}
    assert result["real"].shape == (1, 84)
    assert result["real"][0, 0] == 2
    assert result["real"][0, 21] == 1
    assert not result["lagged"].any() and not result["state"].any()
    a[:, -16:, 0] = False
    assert not contrast_features(v, a, torch.tensor([0.2]))["real"].any()


def test_teaching_keeps_base_peaks_in_every_alarm_band_and_cadence():
    lead = np.r_[
        np.linspace(140.0, 126.0, 5),
        125.0,
        np.linspace(123.0, 31.0, 5),
        30.0,
        np.linspace(29.0, 0.0, 5),
    ]
    risk = np.linspace(-1, 1, len(lead))
    q = chronological_queries(risk, lead, disrupted=True)
    for i in [4, 10, 16]:  # maximum in each distinct lead band
        assert i in q
    assert len(q) == len(set(q)) and (np.diff(q) > 0).all()


def test_pair_gradient_matches_true_interior_displacement(request):
    prior = torch.backends.mha.get_fastpath_enabled()
    request.addfinalizer(lambda: torch.backends.mha.set_fastpath_enabled(prior))
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(13)
    base = (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23),
            torch.ones(23),
        )
        .double()
        .eval()
        .requires_grad_(False)
    )
    x = torch.randn(2, 128, 23, dtype=torch.float64)
    a = torch.ones_like(x, dtype=torch.bool)
    w = torch.zeros((2, 2, 17), dtype=x.dtype)
    w[:, 0, 9:13] = 1.0
    w[:, 1, -1] = 2.0
    z, g = effect_gradient(base, x, a, w)
    assert g.shape == (2, 256)
    with torch.no_grad():
        assert torch.equal(z, base(x, a)["logit"])
        d = g / g.norm(dim=1, keepdim=True)
        plus = routed_pair(base, x, a, w, d * 1e-4)
        minus = routed_pair(base, x, a, w, -d * 1e-4)
    assert torch.allclose((plus - minus) / 2e-4, g.norm(dim=1), rtol=2e-6, atol=1e-9)
    assert all(not block.norm1._forward_hooks for block in base.encoder.layers)


def test_features_require_aligned_finite_signal_shape():
    a = torch.ones((1, 128, 23), dtype=torch.bool)
    with pytest.raises(ValueError, match="shape"):
        contrast_features(torch.zeros((1, 127, 23)), a, torch.zeros(1))


def test_actual_policy_invariance_to_post_boundary_mutation(request):
    from trajectory_replay_20261105 import ARMS, SPEC, predict

    previous = torch.backends.mha.get_fastpath_enabled()
    request.addfinalizer(lambda: torch.backends.mha.set_fastpath_enabled(previous))
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(31)
    base = (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23),
            torch.ones(23),
        )
        .eval()
        .requires_grad_(False)
    )
    x = torch.randn(195, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    mutated = x.clone()
    mutated[167:] *= 20
    mask = a.clone()
    mask[167:, 1] = False
    models = {
        name: (torch.ones(84), torch.randn(84, 256) * 0.02) for name, _, _ in SPEC
    }

    def run(v, m):
        signal = dict(values=v.numpy(), available=m.numpy())
        vx, ma = window_views(v, m)
        with torch.no_grad():
            p = base(vx, ma)["logit"].sigmoid().numpy()
        return predict(base, signal, p, 0.96, models, torch.device("cpu"))[0]

    first, second = run(x, a), run(mutated, mask)
    assert all(np.array_equal(first[k][:167], second[k][:167]) for k in ARMS)
    assert all(np.array_equal(first[k][:135], first["static"][:135]) for k in ARMS)
    assert not np.array_equal(first["trajectory_real"][135:], first["static"][135:])
