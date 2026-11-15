from copy import deepcopy

import pytest
import torch

from experiments.risk_aligned_online.cold20_compare_20261104.adapters import (
    make_adapter,
    named_trainables,
    trainable_parameters,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)


@pytest.fixture(scope="module", autouse=True)
def deterministic_backend():
    old_mha = torch.backends.mha.get_fastpath_enabled()
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_threads = torch.get_num_threads()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    yield
    torch.backends.mha.set_fastpath_enabled(old_mha)
    torch.backends.cuda.matmul.allow_tf32 = old_matmul
    torch.backends.cudnn.allow_tf32 = old_cudnn
    torch.set_num_threads(old_threads)


@pytest.fixture(scope="module")
def qualified_parent():
    # load_parent verifies the qualified checkpoint SHA before loading it.
    base, _ = load_parent(DEFAULT_PARENT)
    return base


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def base(request, qualified_parent):
    return deepcopy(qualified_parent).to(dtype=request.param)


def inputs(base, *, partial):
    generator = torch.Generator(device="cpu").manual_seed(20261104)
    noise = torch.randn(2, 128, 23, generator=generator, dtype=base.center.dtype)
    x = base.center + 0.17 * base.scale * noise
    mask = torch.ones_like(x, dtype=torch.bool)
    if partial:
        mask[:, :97] = False
        mask[:, :, 2] = False
        mask[1, -9:, 1] = False
    return x, mask


def snapshot_source(model):
    return {
        "state": {n: v.detach().clone() for n, v in model.state_dict().items()},
        "flags": {n: p.requires_grad for n, p in model.named_parameters()},
        "grads": {
            n: None if p.grad is None else p.grad.detach().clone()
            for n, p in model.named_parameters()
        },
        "training": {n: module.training for n, module in model.named_modules()},
    }


def assert_source_unchanged(model, before):
    assert set(model.state_dict()) == set(before["state"])
    for name, value in model.state_dict().items():
        assert torch.equal(value, before["state"][name])
    for name, param in model.named_parameters():
        assert param.requires_grad == before["flags"][name]
        old_grad = before["grads"][name]
        assert (param.grad is None) if old_grad is None else torch.equal(param.grad, old_grad)
    assert {n: m.training for n, m in model.named_modules()} == before["training"]


def storage_ids(model):
    return {
        tensor.untyped_storage().data_ptr()
        for tensor in (*model.parameters(), *model.buffers())
        if tensor.numel()
    }


@pytest.mark.parametrize("mode", ["norm", "qv_lora"])
@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial-prefix"])
def test_zero_init_risk_and_forecast_exact_source_parity(base, mode, partial):
    before = snapshot_source(base)
    model = make_adapter(base, mode)
    x, mask = inputs(base, partial=partial)
    with torch.no_grad():
        expected = base(x, mask)
        actual = model(x, mask)
    assert torch.equal(actual["logit"], expected["logit"])
    assert torch.equal(actual["forecast"], expected["forecast"])
    assert not any(module.training for module in model.modules())
    assert not storage_ids(base).intersection(storage_ids(model))
    assert_source_unchanged(base, before)


@pytest.mark.parametrize("mode,count", [("norm", 256), ("qv_lora", 512)])
def test_exact_trainable_scope_and_count(base, mode, count):
    model = make_adapter(base, mode)
    named = dict(named_trainables(model))
    assert len(named) == 4
    assert trainable_parameters(model) == count
    assert sum(p.numel() for p in named.values()) == count
    assert set(named) == {name for name, p in model.named_parameters() if p.requires_grad}
    if mode == "norm":
        assert set(named) == {
            f"encoder.layers.1.norm{number}.{part}"
            for number in (1, 2) for part in ("weight", "bias")
        }
    else:
        assert all("encoder.layers.1.self_attn.parametrizations.in_proj_weight.0." in name for name in named)
        assert sorted(tuple(p.shape) for p in named.values()) == [(2, 64), (2, 64), (64, 2), (64, 2)]
    for param in named.values():
        assert param.dtype == base.center.dtype
        assert param.device == base.center.device
    model.requires_grad_(False)
    assert named_trainables(model) == []
    assert trainable_parameters(model) == 0


@pytest.mark.parametrize("mode", ["norm", "qv_lora"])
def test_source_state_flags_grad_and_buffer_storage_unchanged(base, mode):
    # Check preservation even when the caller already has a stale source grad.
    base.embedding.weight.grad = torch.full_like(base.embedding.weight, 0.125)
    before = snapshot_source(base)
    model = make_adapter(base, mode)
    assert_source_unchanged(base, before)
    assert not storage_ids(base).intersection(storage_ids(model))
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        model.center.add_(1)
    assert_source_unchanged(base, before)


@pytest.mark.parametrize("mode", ["norm", "qv_lora"])
def test_only_authorized_parameters_receive_grad_and_optimizer_changes(base, mode):
    before_source = snapshot_source(base)
    model = make_adapter(base, mode)
    trainables = dict(named_trainables(model))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    optimizer = torch.optim.SGD(list(trainables.values()), lr=0.05)
    x, mask = inputs(base, partial=True)
    out = model(x, mask)
    loss = out["logit"].square().mean() + out["forecast"].square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainables.values())
    assert all(p.grad is None for name, p in model.named_parameters() if name not in trainables)
    optimizer.step()
    changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
    assert changed and changed <= set(trainables)
    assert all(torch.equal(b, buffers[name]) for name, b in model.named_buffers())
    assert_source_unchanged(base, before_source)
    restored = make_adapter(base, mode)
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        expected, actual = model(x, mask), restored(x, mask)
    assert torch.equal(actual["logit"], expected["logit"])
    assert torch.equal(actual["forecast"], expected["forecast"])


def test_lora_qv_only_seed_is_deterministic_and_k_never_changes(base):
    before_rng = torch.random.get_rng_state().clone()
    first = make_adapter(base, "qv_lora")
    second = make_adapter(base, "qv_lora")
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    first_lora = first.encoder.layers[-1].self_attn.parametrizations.in_proj_weight[0]
    second_lora = second.encoder.layers[-1].self_attn.parametrizations.in_proj_weight[0]
    for name in ("lora_A_q", "lora_A_v"):
        value = getattr(first_lora, name)
        assert torch.equal(value, getattr(second_lora, name))
        assert torch.count_nonzero(value) == value.numel()
        assert 0.01 < float(value.detach().std()) < 0.03
    assert not torch.equal(first_lora.lora_A_q, first_lora.lora_A_v)
    for name in ("lora_B_q", "lora_B_v"):
        assert torch.count_nonzero(getattr(first_lora, name)) == 0
    weight = base.encoder.layers[-1].self_attn.in_proj_weight.detach()
    with torch.no_grad():
        first_lora.lora_B_q.fill_(0.1)
        first_lora.lora_B_v.fill_(-0.1)
        updated = first.encoder.layers[-1].self_attn.in_proj_weight
        assert torch.equal(updated[:64], weight[:64] + first_lora.lora_B_q @ first_lora.lora_A_q)
        assert torch.equal(updated[64:128], weight[64:128])
        assert torch.equal(updated[128:], weight[128:] + first_lora.lora_B_v @ first_lora.lora_A_v)
        assert not torch.equal(updated[:64], weight[:64])
        assert not torch.equal(updated[128:], weight[128:])


def test_lora_initialization_uses_source_device_not_global_default(base):
    old_device = torch.get_default_device()
    try:
        # Meta is a CPU-only way to expose accidental reliance on a global
        # default device; the source and adapter themselves remain on CPU.
        torch.set_default_device("meta")
        model = make_adapter(base, "qv_lora")
        assert all(p.device == base.center.device for _, p in named_trainables(model))
    finally:
        torch.set_default_device(old_device)


def test_lora_b_learns_first_and_a_receives_gradient_after_one_step(base):
    model = make_adapter(base, "qv_lora")
    lora = model.encoder.layers[-1].self_attn.parametrizations.in_proj_weight[0]
    optimizer = torch.optim.SGD([p for _, p in named_trainables(model)], lr=0.1)
    x, mask = inputs(base, partial=False)
    before_a = (lora.lora_A_q.detach().clone(), lora.lora_A_v.detach().clone())
    out = model(x, mask)
    (out["logit"].square().mean() + out["forecast"].square().mean()).backward()
    assert lora.lora_B_q.grad.norm() > 0 and lora.lora_B_v.grad.norm() > 0
    assert torch.count_nonzero(lora.lora_A_q.grad) == 0
    assert torch.count_nonzero(lora.lora_A_v.grad) == 0
    optimizer.step()
    assert torch.equal(lora.lora_A_q, before_a[0]) and torch.equal(lora.lora_A_v, before_a[1])
    assert torch.count_nonzero(lora.lora_B_q) > 0 and torch.count_nonzero(lora.lora_B_v) > 0
    optimizer.zero_grad(set_to_none=True)
    out = model(x, mask)
    (out["logit"].square().mean() + out["forecast"].square().mean()).backward()
    assert lora.lora_A_q.grad.norm() > 0 and lora.lora_A_v.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(lora.lora_A_q, before_a[0])
    assert not torch.equal(lora.lora_A_v, before_a[1])


def test_rejects_non_simple23_and_stacked_parametrizations(base):
    with pytest.raises(TypeError):
        make_adapter(torch.nn.Linear(64, 64), "norm")
    adapted = make_adapter(base, "qv_lora")
    before = snapshot_source(adapted)
    for mode in ("norm", "qv_lora"):
        with pytest.raises(ValueError, match="unparametrized"):
            make_adapter(adapted, mode)
    assert_source_unchanged(adapted, before)


@pytest.mark.parametrize("invalid", ["training_submodule", "trainable_source"])
def test_requires_frozen_eval_source_without_changing_its_flags(base, invalid):
    if invalid == "training_submodule":
        base.encoder.layers[0].train()
    else:
        base.risk_head.bias.requires_grad_(True)
    before = snapshot_source(base)
    with pytest.raises(ValueError, match="frozen.*eval"):
        make_adapter(base, "qv_lora")
    assert_source_unchanged(base, before)


@pytest.mark.parametrize("mode", ["", "lora", "Qv_Lora", "head", None])
def test_invalid_mode_rejected_without_changing_source(base, mode):
    before = snapshot_source(base)
    with pytest.raises((TypeError, ValueError)):
        make_adapter(base, mode)
    assert_source_unchanged(base, before)
