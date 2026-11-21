import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Simple23, Config
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import (
    feedback,
    trace,
    window_batch,
)


def data(n=270):
    torch.manual_seed(3)
    b = (
        Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
        .eval()
        .requires_grad_(False)
    )
    rng = np.random.default_rng(4)
    s = dict(
        values=rng.normal(size=(n, 23)).astype("float32"),
        available=np.ones((n, 23), bool),
    )
    return b, s


def test_feedback_rejects_future_targets_and_uses_whole_true_eight_points():
    b, s = data()
    with pytest.raises(ValueError, match="arrived"):
        feedback(b, s, 127, 134, device="cpu")
    k, y, a = feedback(b, s, 127, 135, device="cpu")
    assert k.shape == (1, 70) and y.shape == a.shape == (1, 168)
    cols = [i for i in range(23) if i not in (2, 10)]
    expected = np.arcsinh(s["values"][128:136, cols]) - np.arcsinh(
        s["values"][127, cols]
    )
    np.testing.assert_allclose(y.numpy().reshape(8, 21), expected, atol=2e-7)


def test_prefix_extension_never_changes_existing_innovation():
    b, s = data()
    w = torch.zeros(168, 70)
    full, rows = trace(b, s, w, "real", device="cpu")
    short, _ = trace(b, {k: v[:190] for k, v in s.items()}, w, "real", device="cpu")
    assert torch.equal(full[:190], short)
    assert not full[:135].any() and full[135:].abs().sum() > 0
    assert [r["arrival"] for r in rows] == [135, 167, 199, 231, 263]


def test_fresh_shot_reset_and_discard_are_exact():
    b, s = data()
    w = torch.zeros(168, 70)
    out, _ = trace(b, s, w, "real", device="cpu")
    again, _ = trace(b, s, w, "real", device="cpu")
    zero, _ = trace(b, s, w, "discard", device="cpu")
    assert torch.equal(out, again) and not zero.any() and not w.any()


def test_window_batch_never_uses_values_after_own_endpoint():
    _, s = data()
    x, _ = window_batch(s, [0, 127, 135], device="cpu")
    np.testing.assert_array_equal(x[0, -1].numpy(), s["values"][0])
    assert not x[0, :-1].any()
    np.testing.assert_array_equal(x[1].numpy(), s["values"][:128])
    np.testing.assert_array_equal(x[2].numpy(), s["values"][8:136])


def test_ridge_rule_resets_and_preserves_causal_prefix():
    b, s = data()
    w = torch.zeros(168, 70)
    result, rows = trace(b, s, w, "real", device="cpu", rule="ridge")
    prefix, _ = trace(
        b, {k: v[:190] for k, v in s.items()}, w, "real", device="cpu", rule="ridge"
    )
    assert torch.equal(prefix, result[:190])
    assert rows[-1]["weight_displacement"] <= 2.00001
    discarded, _ = trace(b, s, w, "discard", device="cpu", rule="ridge")
    assert not discarded.any()
