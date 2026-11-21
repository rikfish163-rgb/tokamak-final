import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from motion_metric_model_20261107 import basis_for, encode_windows, forward_state, motion_target, scramble


@pytest.fixture
def fixture():
    old = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(131)
    base = Simple23(Config(fast_site="tokens", fast_rule="normalized"),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(3, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    a[1, :32] = False
    a[2, :, 2] = False
    yield base, x, a
    torch.backends.mha.set_fastpath_enabled(old)


def test_zero_internal_state_preserves_both_native_heads(fixture):
    base, x, a = fixture
    with torch.no_grad():
        expected = base(x, a)
    pack = encode_windows(base, x, a, basis_for(base))
    assert pack["risk_hidden"].shape == (len(x), 64)
    torch.testing.assert_close(base.risk_head(pack["risk_hidden"]).squeeze(-1), pack["native"], atol=0., rtol=0.)
    got = forward_state(base, pack, torch.zeros(3, 4, 64))
    for key in ("logit", "forecast"):
        torch.testing.assert_close(got[key], expected[key], rtol=1e-5, atol=2e-6)


def test_actual_ffn_hook_matches_cached_nonzero_neural_update(fixture):
    base, x, a = fixture
    basis = basis_for(base)
    pack = encode_windows(base, x, a, basis)
    w = torch.randn(3, 4, 64) * .03
    got = forward_state(base, pack, w)
    def hook(_module, args, output):
        from experiments.risk_aligned_online.lora_stream_20261102.core import cap_shift
        shift, _ = cap_shift((args[0][:, :16] @ basis.T / 2) @ w, pack["valid"][:, :16])
        return output + torch.cat((shift, torch.zeros_like(output[:, -1:])), 1)
    handle = base.encoder.layers[0].linear2.register_forward_hook(hook)
    try:
        expected = base(x, a)
    finally:
        handle.remove()
    for key in ("logit", "forecast"):
        torch.testing.assert_close(got[key], expected[key], rtol=1e-5, atol=3e-6)


def test_target_contains_arrived_changes_and_masks_unobserved_origin(fixture):
    base, x, a = fixture
    x[:, -1] = .5
    future = torch.full((3, 25, 23), 1.)
    mask = torch.ones_like(future, dtype=torch.bool)
    a[0, -1, 5] = False
    motion, valid = motion_target(base, x, a, future, mask)
    assert not valid[:, :, [2, 10]].any()
    assert not valid[0, :, 5].any()
    expected = torch.asinh(torch.tensor(1.)) - torch.asinh(torch.tensor(.5))
    torch.testing.assert_close(motion[valid], torch.full_like(motion[valid], expected))
    assert (motion[~valid] == 0).all()


def test_controls_keep_the_actual_observed_support_and_value_multiset():
    torch.manual_seed(131)
    x = torch.randn(2, 25, 23)
    mask = torch.rand_like(x) > .2
    for axis in ("time", "channel"):
        transformed = scramble(x.masked_fill(~mask, 0), mask, axis=axis)
        assert (transformed[~mask] == 0).all()
        original = x.transpose(1, 2) if axis == "time" else x
        actual = transformed.transpose(1, 2) if axis == "time" else transformed
        observed = mask.transpose(1, 2) if axis == "time" else mask
        for a, b, m in zip(original.flatten(0, 1), actual.flatten(0, 1), observed.flatten(0, 1)):
            torch.testing.assert_close(a[m].sort().values, b[m].sort().values)


def test_actual_source_meta_gradient_reaches_metric_without_changing_source(fixture):
    from motion_metric_kernel_20261107 import MotionMetric
    from run_motion_metric_20261107 import state

    base, x, a = fixture
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    origin = encode_windows(base, x, a, basis_for(base))
    query = encode_windows(base, x.roll(5, 1), a, basis_for(base))
    metric = MotionMetric()
    context = torch.randn(3, 69)
    motion = torch.randn(3, 25, 23)
    mask = torch.ones_like(motion, dtype=torch.bool)
    updated, _ = state(base, origin, context, motion, mask, metric, create_graph=True)
    risk = forward_state(base, query, updated)["logit"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(risk, torch.tensor([1., 0., 1.]))
    gradients = torch.autograd.grad(loss, tuple(metric.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0
    assert all(torch.equal(frozen[k], v) for k, v in base.state_dict().items())
    assert all(p.grad is None for p in base.parameters())
