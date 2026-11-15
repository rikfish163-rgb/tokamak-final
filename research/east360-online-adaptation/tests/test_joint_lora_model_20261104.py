import gc
import weakref
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from experiments.risk_aligned_online.direct_fast_memory_20261104.memory import (
    dynamic_key,
)
from experiments.risk_aligned_online.direct_fast_memory_20261104.model import levels
from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)


@pytest.fixture(scope="module", autouse=True)
def deterministic_cpu_backend():
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
def parent():
    # This loader verifies the qualified checkpoint hash before loading.
    base, _ = load_parent(DEFAULT_PARENT)
    return base


@pytest.fixture
def base(parent):
    return deepcopy(parent)


@pytest.fixture
def initial():
    return (
        torch.randn(168, 70, generator=torch.Generator().manual_seed(20261104)) * 0.02
    )


def inputs(base, partial=False):
    noise = torch.randn(2, 128, 23, generator=torch.Generator().manual_seed(401))
    x = base.center + 0.2 * base.scale * noise
    available = torch.ones_like(x, dtype=torch.bool)
    if partial:
        available[:, :91] = False
        available[:, :, 2] = False
        available[1, -10:, 1] = False
    return x, available


def snapshot(module):
    return dict(
        state={k: v.detach().clone() for k, v in module.state_dict().items()},
        flags={k: p.requires_grad for k, p in module.named_parameters()},
        grads={
            k: None if p.grad is None else p.grad.detach().clone()
            for k, p in module.named_parameters()
        },
        modes={k: m.training for k, m in module.named_modules()},
    )


def assert_same_source(module, before):
    assert module.state_dict().keys() == before["state"].keys()
    for name, value in module.state_dict().items():
        assert torch.equal(value, before["state"][name])
    for name, parameter in module.named_parameters():
        assert parameter.requires_grad == before["flags"][name]
        old = before["grads"][name]
        assert (
            (parameter.grad is None)
            if old is None
            else torch.equal(parameter.grad, old)
        )
    assert {k: m.training for k, m in module.named_modules()} == before["modes"]


def storage_ids(tensors):
    return {t.untyped_storage().data_ptr() for t in tensors if t.numel()}


@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial-prefix"])
def test_qualified_zero_init_exact_risk_and_forecast_parity(base, initial, partial):
    before = snapshot(base)
    model = JointLoRA(base, initial)
    x, available = inputs(base, partial)
    with torch.no_grad():
        source = base(x, available)
        result = model(x, available)
        native = model.clone(x, available)
    assert set(result) == {"logit", "diagnostic", "key", "latent"}
    assert torch.equal(result["logit"], source["logit"])
    assert torch.equal(native["forecast"], source["forecast"])
    expected_key = dynamic_key(levels(model.clone, x, available), available)
    assert torch.equal(result["key"], expected_key)
    assert torch.equal(result["diagnostic"], expected_key @ model.W0.T)
    assert result["diagnostic"].shape == (2, 168)
    assert result["latent"].shape == (2, 192)
    assert torch.count_nonzero(model.residual_decoder.weight) == 0
    assert torch.count_nonzero(model.residual_decoder.bias) == 0
    assert_same_source(base, before)


def test_real_norm_tokens_order_one_source_call_and_scoped_hook(base, initial):
    model = JointLoRA(base, initial)
    x, available = inputs(base, partial=True)
    observations = []
    handle = model.clone.norm.register_forward_hook(
        lambda _, __, out: observations.append(out)
    )
    try:
        old_hooks = set(model.clone.norm._forward_hooks)
        with patch.object(model.clone, "forward", wraps=model.clone.forward) as call:
            actual = model(x, available)
        assert call.call_count == 1
        assert len(observations) == 1
        assert set(model.clone.norm._forward_hooks) == old_hooks
        normed = observations[0]
        assert normed.shape == (2, 17, 64)
        expected = torch.cat(
            (normed[:, -1], normed[:, -2], normed[:, -2] - normed[:, -3]), dim=1
        )
        assert torch.equal(actual["latent"], expected)
    finally:
        handle.remove()


def test_missing_last_patches_uses_raw_norm_hook_not_postmask_tokens(base, initial):
    model = JointLoRA(base, initial)
    x, available = inputs(base)
    available[:, -16:] = False
    observations = []
    handle = model.clone.norm.register_forward_hook(
        lambda _module, _args, output: observations.append(output.detach())
    )
    try:
        with torch.no_grad():
            result = model(x, available)
            expected_logit = base(x, available)["logit"]
    finally:
        handle.remove()
    assert torch.equal(result["logit"], expected_logit)
    assert torch.equal(result["latent"][:, 64:128], observations[0][:, -2])
    assert torch.count_nonzero(result["latent"][:, 64:128]) > 0


def test_hook_removed_on_failure_before_or_after_norm(base, initial):
    model = JointLoRA(base, initial)
    x, available = inputs(base)
    old_hooks = dict(model.clone.norm._forward_hooks)
    with pytest.raises(ValueError):
        model(x, available.float())
    assert dict(model.clone.norm._forward_hooks) == old_hooks
    with patch.object(
        model.clone.forecast_head, "forward", side_effect=RuntimeError("after norm")
    ):
        with pytest.raises(RuntimeError, match="after norm"):
            model(x, available)
    assert dict(model.clone.norm._forward_hooks) == old_hooks
    assert model(x, available)["latent"].shape == (2, 192)
    assert dict(model.clone.norm._forward_hooks) == old_hooks


def test_decoder_cannot_change_risk_or_key(base, initial):
    model = JointLoRA(base, initial)
    x, available = inputs(base, partial=True)
    first = model(x, available)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(99)
        model.residual_decoder.weight.copy_(
            torch.randn(168, 192, generator=generator) * 0.02
        )
        model.residual_decoder.bias.fill_(0.1)
    second = model(x, available)
    assert torch.equal(first["logit"], second["logit"])
    assert torch.equal(first["key"], second["key"])
    assert not torch.equal(first["diagnostic"], second["diagnostic"])


def test_zero_decoder_first_diagnostic_gradient_only_reaches_decoder(base, initial):
    model = JointLoRA(base, initial)
    x, available = inputs(base)
    model(x, available)["diagnostic"].square().mean().backward()
    lora = model.clone.encoder.layers[-1].self_attn.parametrizations.in_proj_weight[0]
    assert model.residual_decoder.weight.grad.norm() > 0
    assert torch.count_nonzero(lora.lora_B_q.grad) == 0
    assert torch.count_nonzero(lora.lora_B_v.grad) == 0


def test_temporary_hook_does_not_retain_forward_graph(base, initial):
    model = JointLoRA(base, initial)
    x, available = inputs(base)
    refs = []
    observer = model.clone.norm.register_forward_hook(
        lambda _module, _args, output: refs.append(weakref.ref(output))
    )
    try:
        result = model(x, available)
    finally:
        observer.remove()
    assert len(refs) == 1
    del result
    gc.collect()
    assert refs[0]() is None
    assert len(model.clone.norm._forward_hooks) == 0


@pytest.mark.parametrize("partial", [False, True])
def test_zero_init_risk_exact_with_autograd_enabled(base, initial, partial):
    model = JointLoRA(base, initial)
    x, available = inputs(base, partial)
    assert torch.equal(model(x, available)["logit"], base(x, available)["logit"])


def test_counts_scopes_and_independent_source_and_w0(base, initial):
    base.embedding.weight.grad = torch.full_like(base.embedding.weight, 0.125)
    before = snapshot(base)
    initial_snapshot = initial.clone()
    model = JointLoRA(base, initial)
    neural = model.neural_parameters()
    decoder = model.decoder_parameters()
    assert sum(p.numel() for p in neural) == 512
    assert sum(p.numel() for p in decoder) == 32424
    assert len(neural) == 4 and len(decoder) == 2
    assert {id(p) for p in model.parameters() if p.requires_grad} == {
        id(p) for p in (*neural, *decoder)
    }
    names = [
        name for name, p in model.named_parameters() if id(p) in {id(q) for q in neural}
    ]
    assert all(
        "encoder.layers.1.self_attn.parametrizations.in_proj_weight.0.lora_" in n
        for n in names
    )
    assert not model.W0.requires_grad and "W0" in dict(model.named_buffers())
    assert model.W0.data_ptr() != initial.data_ptr()
    source_storage = storage_ids((*base.parameters(), *base.buffers()))
    model_storage = storage_ids((*model.parameters(), *model.buffers()))
    assert not source_storage.intersection(model_storage)
    with torch.no_grad():
        initial.fill_(4.0)
    assert torch.equal(model.W0, initial_snapshot)
    for parameter in neural:
        parameter.requires_grad_(False)
    assert {id(p) for p in model.neural_parameters()} == {id(p) for p in neural}
    assert_same_source(base, before)


def test_risk_gradient_and_optimizer_step_only_lora_and_not_source(base, initial):
    before_source = snapshot(base)
    model = JointLoRA(base, initial)
    before_clone = {
        name: p.detach().clone() for name, p in model.clone.named_parameters()
    }
    before_w0 = model.W0.clone()
    neural = model.neural_parameters()
    x, available = inputs(base, partial=True)
    optimizer = torch.optim.SGD(neural, lr=0.05)
    model(x, available)["logit"].square().mean().backward()
    lora = model.clone.encoder.layers[-1].self_attn.parametrizations.in_proj_weight[0]
    assert lora.lora_B_q.grad.norm() > 0 and lora.lora_B_v.grad.norm() > 0
    assert all(p.grad is None for p in model.decoder_parameters())
    assert all(
        p.grad is None
        for p in model.clone.parameters()
        if id(p) not in {id(q) for q in neural}
    )
    optimizer.step()
    changed = {
        name
        for name, p in model.clone.named_parameters()
        if not torch.equal(p, before_clone[name])
    }
    allowed = {
        name
        for name, p in model.clone.named_parameters()
        if id(p) in {id(q) for q in neural}
    }
    assert changed and changed <= allowed
    assert torch.equal(model.W0, before_w0)
    assert_same_source(base, before_source)


def test_nonzero_diagnostic_decoder_backpropagates_to_lora(base, initial):
    before = snapshot(base)
    model = JointLoRA(base, initial)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(991)
        model.residual_decoder.weight.copy_(
            torch.randn(168, 192, generator=generator) * 0.02
        )
    x, available = inputs(base)
    model(x, available)["diagnostic"].square().mean().backward()
    lora = model.clone.encoder.layers[-1].self_attn.parametrizations.in_proj_weight[0]
    assert lora.lora_B_q.grad.norm() > 0 and lora.lora_B_v.grad.norm() > 0
    assert model.residual_decoder.weight.grad.norm() > 0
    assert all(p.grad is None for p in model.clone.parameters() if not p.requires_grad)
    assert_same_source(base, before)


def test_train_keeps_pretrained_clone_in_eval_but_gradients_enabled(base, initial):
    model = JointLoRA(base, initial)
    assert not any(module.training for module in model.clone.modules())
    model.train()
    assert not any(module.training for module in model.clone.modules())
    assert model.residual_decoder.training
    assert all(p.requires_grad for p in model.neural_parameters())
    model.eval()
    assert not model.residual_decoder.training


def test_state_dict_round_trip_preserves_w0_lora_and_decoder(base, initial):
    model = JointLoRA(base, initial)
    with torch.no_grad():
        lora = model.clone.encoder.layers[-1].self_attn.parametrizations.in_proj_weight[
            0
        ]
        lora.lora_B_q.fill_(0.01)
        lora.lora_B_v.fill_(-0.02)
        model.residual_decoder.weight.fill_(0.003)
        model.residual_decoder.bias.fill_(0.04)
    restored = JointLoRA(base, torch.zeros_like(initial))
    restored.load_state_dict(deepcopy(model.state_dict()), strict=True)
    x, available = inputs(base, partial=True)
    with torch.no_grad():
        expected, actual = model(x, available), restored(x, available)
    for key in ("logit", "diagnostic", "key", "latent"):
        assert torch.equal(actual[key], expected[key])
    assert torch.equal(restored.W0, initial)
    assert restored.W0.data_ptr() != model.W0.data_ptr()


@pytest.mark.parametrize(
    "changes",
    [
        {"window": 64},
        {"patch": 16},
        {"dim": 32},
        {"layers": 1},
        {"heads": 8},
        {"horizon": 8},
        {"fast_site": "output"},
        {"fast_rule": "raw"},
        {"input_view": "hybrid"},
    ],
)
def test_rejects_wrong_source_configuration_without_mutation(base, initial, changes):
    base.cfg = replace(base.cfg, **changes)
    before = snapshot(base)
    with pytest.raises(ValueError):
        JointLoRA(base, initial)
    assert_same_source(base, before)


@pytest.mark.parametrize("case", ["trainable", "submodule_training", "dtype"])
def test_rejects_unfrozen_non_eval_or_wrong_dtype_source(base, initial, case):
    if case == "trainable":
        base.risk_head.bias.requires_grad_(True)
    elif case == "submodule_training":
        base.encoder.layers[0].train()
    else:
        base.double()
    before = snapshot(base)
    with pytest.raises(ValueError):
        JointLoRA(base, initial)
    assert_same_source(base, before)


@pytest.mark.parametrize(
    "case",
    [
        "center_shape",
        "scale_shape",
        "zero_scale",
        "embedding",
        "risk",
        "forecast",
        "heads",
    ],
)
def test_rejects_mutated_source_geometry_or_scaler(base, initial, case):
    if case == "center_shape":
        base.center = torch.zeros(22)
    elif case == "scale_shape":
        base.scale = torch.ones(22)
    elif case == "zero_scale":
        base.scale[0] = 0
    elif case == "embedding":
        base.embedding = torch.nn.Linear(367, 64).requires_grad_(False).eval()
    elif case == "risk":
        base.risk_head = torch.nn.Linear(64, 2).requires_grad_(False).eval()
    elif case == "forecast":
        base.forecast_head = torch.nn.Linear(1088, 184).requires_grad_(False).eval()
    else:
        base.encoder.layers[0].self_attn.num_heads = 8
    before = snapshot(base)
    with pytest.raises(ValueError):
        JointLoRA(base, initial)
    assert_same_source(base, before)


@pytest.mark.parametrize("case", ["shape", "float64", "nan", "inf", "list"])
def test_rejects_invalid_w0_without_touching_source(base, initial, case):
    if case == "shape":
        initial = initial[:, :-1]
    elif case == "float64":
        initial = initial.double()
    elif case == "nan":
        initial[0, 0] = float("nan")
    elif case == "inf":
        initial[0, 0] = float("inf")
    else:
        initial = []
    before = snapshot(base)
    with pytest.raises((ValueError, TypeError)):
        JointLoRA(base, initial)
    assert_same_source(base, before)


def test_w0_detached_clone_and_finite_input_mask_contract(base, initial):
    initial.requires_grad_(True)
    model = JointLoRA(base, initial)
    x, available = inputs(base)
    x[0, 0, 1] = float("nan")
    available[0, 0, 1] = False
    output = model(x, available)
    assert torch.isfinite(output["diagnostic"]).all()
    output["diagnostic"].sum().backward()
    assert initial.grad is None and not model.W0.requires_grad
    available[0, 0, 1] = True
    with pytest.raises(ValueError, match="finite"):
        model(x, available)
