"""Contract tests for the literature-inspired variate-token alternative."""
import pytest
import torch

from experiments.risk_aligned_online.model import ModelConfig, RiskAlignedModel


def make_model():
    torch.manual_seed(13)
    cfg = ModelConfig(backbone='variate', window=16, patch=8, dim=16,
                      heads=2, rank=4, horizon=5)
    return RiskAlignedModel(cfg, torch.zeros(23), torch.ones(23))


def test_channel_tokens_direct_forecast_and_fast_gradient():
    model = make_model()
    x = torch.randn(2, 16, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    fast = model.initial_fast(2)
    out = model(x, mask, fast=fast)
    assert out['logits'].shape == (2, 2)
    assert out['forecast'].shape == (2, 5, 23)
    assert model.position.shape[1] == 24  # 23 diagnostic tokens plus risk CLS
    grad, = torch.autograd.grad(out['forecast'].square().mean(), fast, create_graph=True)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    changed = model(x, mask, fast=fast - .1 * grad)
    assert not torch.equal(out['logits'], changed['logits'])
    changed['logits'].square().mean().backward()
    assert model.adapter_down.weight.grad is not None


def test_masked_values_do_not_change_predictions_and_empty_input_finite():
    model = make_model().eval()
    x = torch.randn(2, 16, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, 3] = False
    altered = x.clone()
    altered[:, :, 3] = float('nan')
    for name in ('logits', 'forecast'):
        torch.testing.assert_close(model(x, mask)[name], model(altered, mask)[name])
        assert torch.isfinite(model(x, torch.zeros_like(mask))[name]).all()


def test_backbone_config_roundtrip_and_legacy_default():
    model = make_model()
    assert ModelConfig(**model.cfg.to_dict()) == model.cfg
    assert ModelConfig().backbone == 'patch'
    with pytest.raises(ValueError, match='backbone'):
        ModelConfig(backbone='unknown')
