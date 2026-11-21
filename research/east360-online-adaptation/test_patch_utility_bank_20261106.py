import numpy as np
import torch

import run_patch_utility_bank_20261106 as study
from experiments.risk_aligned_online.simple23 import Config, Simple23
from run_patch_utility_bank_20261106 import axes, risk_context


def test_bank_axes_ignore_truth_reversal_and_remove_reached_risk_directions():
    rng = np.random.default_rng(13)
    g = rng.normal(size=(3, 2, 9, 64))
    guard = rng.normal(size=(3, 3, 64))
    other = rng.normal(size=(3, 3, 64))
    fit = axes(guard, other, g, np.array([True, True, False]))
    np.testing.assert_array_equal(
        fit, axes(guard, other, g[:, ::-1], np.array([True, True, False]))
    )
    np.testing.assert_allclose(fit[:2] @ guard[:2].transpose(0, 2, 1), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(fit[:2], axis=2), 0.35)
    assert not fit[2].any()


def test_risk_context_never_looks_after_maturity():
    z = np.linspace(-1, 2, 112)
    m = np.array([63, 95])
    old = risk_context(z, m)
    z[96:] = 1e5
    np.testing.assert_array_equal(risk_context(z, m), old)


def test_causal_bank_preserves_zero_action_and_historical_prefix():
    with torch.random.fork_rng():
        torch.manual_seed(913)
        base = (
            Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                torch.zeros(23),
                torch.ones(23),
            )
            .eval()
            .requires_grad_(False)
        )
    rng = np.random.default_rng(117)
    x = rng.normal(size=(135, 23)).astype(np.float32)
    raw = dict(values=x, available=np.isfinite(x), times=np.arange(len(x)) * 0.001)
    projection = study.op.basis(base)
    observed = study.generate(base, projection, raw)
    assert observed["allowed"][:, 0].all()
    assert observed["allowed"][:, 1:].any()
    np.testing.assert_array_equal(
        observed["allowed"][:, 1::2], observed["allowed"][:, 2::2]
    )
    assert np.abs(observed["effect"][..., :3]).max() <= 0.1
    for j, m in enumerate(observed["clock"]):
        q = observed["native"][m + 1 : m + 33]
        np.testing.assert_array_equal(q, observed["curves"][j, 0, : len(q)])
    assert max(study.prefix_check(base, projection, raw, observed, 119).values()) < 8e-5
