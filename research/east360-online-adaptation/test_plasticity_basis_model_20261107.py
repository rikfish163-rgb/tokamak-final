import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.lora_stream_20261102.core import cap_shift
from experiments.risk_aligned_online.simple23 import Config, Simple23
import motion_metric_model_20261107 as motion_model
from motion_metric_kernel_20261107 import magnitude_descent, potential
import plasticity_basis_model_20261107 as model


@pytest.fixture
def sample():
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        base = Simple23(Config(fast_site="tokens", fast_rule="normalized"),
                        torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
        x = torch.randn(3, 128, 23)
        available = torch.ones_like(x, dtype=torch.bool)
        available[1, :32] = False
        available[2, :, 2] = False
        x[2, 0, 4] = float("nan")
        try:
            yield base, x, available
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)


def orthogonal_rows(seed, *, dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(128, 4, dtype=dtype, generator=generator)
    return torch.linalg.qr(values, mode="reduced").Q.T.contiguous()


@pytest.mark.parametrize("seed", [11, 19])
def test_zero_state_preserves_both_native_heads_for_arbitrary_basis(sample, seed):
    base, x, available = sample
    basis = orthogonal_rows(seed)
    with torch.no_grad():
        native = base(x, available)
    pack = model.encode_windows(base, x, available, basis)
    old_pack = motion_model.encode_windows(base, x, available, basis)

    assert set(pack) == set(old_pack) | {"activation"}
    assert pack["activation"].shape == (3, 16, 128)
    assert not pack["activation"].requires_grad
    for key in old_pack:
        torch.testing.assert_close(pack[key], old_pack[key], rtol=0, atol=0)
    result = model.forward_state(base, pack, torch.zeros(3, 4, 64), basis)
    for key in ("logit", "forecast"):
        torch.testing.assert_close(result[key], native[key], rtol=1e-5, atol=2e-6)


def test_source_svd_basis_preserves_nonzero_legacy_forward(sample):
    base, x, available = sample
    basis = motion_model.basis_for(base)
    pack = model.encode_windows(base, x, available, basis)
    legacy_pack = motion_model.encode_windows(base, x, available, basis)
    state = torch.randn(3, 4, 64) * .2
    result = model.forward_state(base, pack, state, basis)
    expected = motion_model.forward_state(base, legacy_pack, state)
    for key in ("logit", "forecast", "rms"):
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)


def test_reprojects_current_basis_without_mutating_cached_pack(sample):
    base, x, available = sample
    first, second = orthogonal_rows(11), orthogonal_rows(29)
    pack = model.encode_windows(base, x, available, first)
    original_phi = pack["phi"].clone()
    state = torch.randn(3, 4, 64) * .3
    result = model.forward_state(base, pack, state, second)
    expected_pack = {**pack, "phi": pack["activation"] @ second.T / 2}
    expected = motion_model.forward_state(base, expected_pack, state)
    stale = motion_model.forward_state(base, pack, state)

    for key in ("logit", "forecast", "rms"):
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)
    assert not torch.allclose(result["forecast"], stale["forecast"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pack["phi"], original_phi, rtol=0, atol=0)


def test_orthogonal_coordinate_rotation_preserves_both_predictions(sample):
    base, x, available = sample
    basis = orthogonal_rows(17)
    rotation = torch.linalg.qr(torch.randn(4, 4)).Q
    state = torch.randn(3, 4, 64) * .4
    pack = model.encode_windows(base, x, available, basis)

    original = model.forward_state(base, pack, state, basis)
    rotated = model.forward_state(base, pack, rotation @ state, rotation @ basis)
    for key in ("logit", "forecast", "rms"):
        torch.testing.assert_close(rotated[key], original[key], rtol=2e-5, atol=3e-6)


def test_actual_linear2_hook_matches_cached_forward_for_nonzero_state(sample):
    base, x, available = sample
    basis = orthogonal_rows(23)
    pack = model.encode_windows(base, x, available, basis)
    state = torch.randn(3, 4, 64) * 2
    cached = model.forward_state(base, pack, state, basis)
    seen = []

    def insert(_module, args, output):
        seen.append(args[0][:, :16].detach().clone())
        shift, _ = cap_shift((args[0][:, :16] @ basis.T / 2) @ state,
                             pack["valid"][:, :16])
        return output + torch.cat((shift, torch.zeros_like(output[:, -1:])), dim=1)

    handle = base.encoder.layers[0].linear2.register_forward_hook(insert)
    try:
        with torch.no_grad():
            direct = base(x, available)
    finally:
        handle.remove()
    assert len(seen) == 1
    torch.testing.assert_close(pack["activation"], seen[0], rtol=0, atol=0)
    for key in ("logit", "forecast"):
        torch.testing.assert_close(cached[key], direct[key], rtol=1e-5, atol=3e-6)


def test_nonzero_state_preserves_direct_basis_gradient(sample):
    base, x, available = sample
    basis = orthogonal_rows(29).requires_grad_()
    pack = model.encode_windows(base, x, available, basis)
    state = torch.randn(3, 4, 64) * .1
    loss = model.forward_state(base, pack, state, basis)["logit"].square().sum()
    gradient, = torch.autograd.grad(loss, basis)
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("initial_level", [0., 1e-6])
def test_risk_gradient_reaches_basis_through_actual_motion_update(sample, initial_level):
    base, x, available = sample
    frozen = {name: value.clone() for name, value in base.state_dict().items()}
    basis = orthogonal_rows(31).requires_grad_()
    origin = model.encode_windows(base, x, available, basis)
    query = model.encode_windows(base, x.roll(5, 1), available, basis)
    future = x[:, -1:, :] + .1 * torch.randn(3, 25, 23)
    future_available = torch.ones_like(future, dtype=torch.bool)
    motion, mask = motion_model.motion_target(base, x, available, future, future_available)

    initial = torch.full((3, 4, 64), initial_level, requires_grad=True)
    forecast = model.forward_state(base, origin, initial, basis)["forecast"]
    motion_loss = potential(forecast, motion, mask, torch.ones(3, 23))
    gradient, = torch.autograd.grad(motion_loss.sum(), initial, create_graph=True)
    updated = magnitude_descent(gradient, reference_norm=.01)
    # Detach the query basis so the only route back is through the inner update.
    risk = model.forward_state(base, query, updated, basis.detach())["logit"]
    risk_loss = F.binary_cross_entropy_with_logits(risk, torch.tensor([1., 0., 1.]))
    state_gradient, = torch.autograd.grad(risk_loss, updated, retain_graph=True)
    basis_gradient, = torch.autograd.grad(risk_loss, basis)

    for value in (gradient, updated, state_gradient, basis_gradient):
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) > 0
    assert all(torch.equal(frozen[name], value) for name, value in base.state_dict().items())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in base.parameters())


def test_qr_reprojection_is_in_place_preserves_rowspace_and_positive_diagonal():
    generator = torch.Generator(device="cpu").manual_seed(71)
    basis = torch.randn(4, 128, generator=generator, dtype=torch.float64).requires_grad_()
    original = basis.detach().clone()
    pointer = basis.data_ptr()
    prior_gradient = torch.full_like(basis, 3.)
    basis.grad = prior_gradient.clone()

    result = model.project_rows(basis)

    assert result is basis
    assert basis.data_ptr() == pointer and basis.is_leaf and basis.requires_grad
    assert basis.grad_fn is None
    torch.testing.assert_close(basis.grad, prior_gradient, rtol=0, atol=0)
    torch.testing.assert_close(basis @ basis.T, torch.eye(4, dtype=basis.dtype), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close((original @ basis.T) @ basis, original, rtol=1e-12, atol=1e-12)
    r = basis @ original.T
    assert (r.diagonal() > 0).all()
    torch.testing.assert_close(r.tril(-1), torch.zeros_like(r), rtol=0, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_random_basis_is_deterministic_orthogonal_and_rng_isolated(dtype):
    reference = torch.zeros(4, 128, dtype=dtype)
    before = torch.random.get_rng_state().clone()
    first = model.random_basis(reference)
    repeat = model.random_basis(reference, seed=131)
    other = model.random_basis(reference, seed=132)
    after = torch.random.get_rng_state()

    assert torch.equal(before, after)
    assert first.shape == reference.shape and first.dtype == dtype and first.device == reference.device
    torch.testing.assert_close(first, repeat, rtol=0, atol=0)
    assert not torch.equal(first, other)
    torch.testing.assert_close(first @ first.T, torch.eye(4, dtype=dtype), rtol=1e-6, atol=1e-6)


def test_capture_only_first_actual_linear2_input_and_remove_hook(sample, monkeypatch):
    base, x, available = sample
    basis = orthogonal_rows(37)
    layer = base.encoder.layers[0].linear2
    original_forward = base.forward
    inputs = []

    def observe(_module, args):
        inputs.append(args[0][:, :16].detach().clone())

    observer = layer.register_forward_pre_hook(observe)
    before_hooks = dict(layer._forward_pre_hooks)

    def two_passes(values, mask):
        first = original_forward(values, mask)
        original_forward(values.roll(13, 1), mask)
        return first

    monkeypatch.setattr(base, "forward", two_passes)
    try:
        pack = model.encode_windows(base, x.requires_grad_(), available, basis)
        assert dict(layer._forward_pre_hooks) == before_hooks
    finally:
        observer.remove()
    assert len(inputs) == 2
    assert not torch.allclose(inputs[0], inputs[1])
    torch.testing.assert_close(pack["activation"], inputs[0], rtol=0, atol=0)
    assert not pack["activation"].requires_grad


def test_capture_hook_is_removed_when_original_forward_raises(sample, monkeypatch):
    base, x, available = sample
    layer = base.encoder.layers[0].linear2
    before_pre_hooks = dict(layer._forward_pre_hooks)
    before_hooks = dict(layer._forward_hooks)
    original_forward = base.forward

    def fail_after_capture(values, mask):
        original_forward(values, mask)
        raise RuntimeError("synthetic encoder failure")

    monkeypatch.setattr(base, "forward", fail_after_capture)
    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        model.encode_windows(base, x, available, orthogonal_rows(41))
    assert dict(layer._forward_pre_hooks) == before_pre_hooks
    assert dict(layer._forward_hooks) == before_hooks


def test_capture_hook_is_removed_when_source_is_not_frozen_and_eval(sample):
    base, x, available = sample
    layer = base.encoder.layers[0].linear2
    before_hooks = dict(layer._forward_pre_hooks)
    base.train()
    with pytest.raises(ValueError, match="frozen and eval"):
        model.encode_windows(base, x, available, orthogonal_rows(43))
    assert dict(layer._forward_pre_hooks) == before_hooks


@pytest.mark.parametrize("operation", ["encode_windows", "forward_state", "project_rows", "random_basis"])
def test_only_experiment_basis_shape_is_supported(sample, operation):
    base, x, available = sample
    wrong = torch.zeros(5, 128)
    with pytest.raises(ValueError, match="4x128"):
        if operation == "encode_windows":
            model.encode_windows(base, x, available, wrong)
        elif operation == "forward_state":
            model.forward_state(base, {}, torch.zeros(3, 4, 64), wrong)
        elif operation == "project_rows":
            model.project_rows(wrong)
        else:
            model.random_basis(wrong)
