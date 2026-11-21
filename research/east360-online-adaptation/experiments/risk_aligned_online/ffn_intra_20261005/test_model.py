import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel


@pytest.fixture
def case():
    torch.set_num_threads(2)
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23)).eval()
    model = FfnModel(base)
    x = torch.randn(2, 128, 23)
    a = torch.rand(x.shape) > .2
    return model, x, a


def test_zero_matches_grad_dispatch_and_fixed_parameters(case):
    model, x, a = case
    before = {k: v.clone() for k, v in model.state_dict().items()}
    delta = model.initial_delta()
    out = model(x, a, delta)
    expected = model.base(x, a, model.base.initial_fast(2).detach().requires_grad_())
    for k in out:
        torch.testing.assert_close(out[k], expected[k], rtol=0, atol=0)
    assert delta.shape == (16576,)
    assert model.parameter_names == ('encoder.layers.1.linear1.weight',
        'encoder.layers.1.linear1.bias', 'encoder.layers.1.linear2.weight',
        'encoder.layers.1.linear2.bias')
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert not model.initial_delta().any()


def test_double_backward_updates_only_external_delta(case):
    model, x, a = case
    delta = model.initial_delta()
    coefficient = torch.tensor(1., requires_grad=True)
    inner = coefficient * model(x, a, delta)['forecast'].square().mean()
    grad, = torch.autograd.grad(inner, delta, create_graph=True)
    updated = delta - .1 * grad
    outer = model(x * .9, a, updated)['logit'].square().mean()
    meta, = torch.autograd.grad(outer, coefficient)
    assert torch.isfinite(meta) and meta.abs() > 0
    assert grad.norm() > 0 and all(p.grad is None for p in model.parameters())
    assert not model.initial_delta().any()


def test_late_delta_keeps_early_attention_differentiable(case):
    model, x, a = case
    seen = []
    hook = model.base.encoder.layers[0].self_attn.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].requires_grad))
    try:
        model(x, a, model.initial_delta())
    finally:
        hook.remove()
    assert seen == [True]


def test_shape_rejected_and_inference_context_explicit(case):
    model, x, a = case
    with pytest.raises(ValueError):
        model(x, a, torch.zeros(2, 16576))
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(x, a, model.initial_delta())


def test_exception_restores_base_parameter_objects(case):
    model, x, a = case
    originals = dict(model.base.named_parameters())
    def fail(*args):
        raise RuntimeError('intentional')
    hook = model.base.encoder.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match='intentional'):
            model(x, a, model.initial_delta())
    finally:
        hook.remove()
    for name, p in model.base.named_parameters():
        assert p is originals[name]
