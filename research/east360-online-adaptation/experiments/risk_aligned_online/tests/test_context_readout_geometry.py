import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.geometry import (
    split_first_direction, evaluate_geometry,
)
from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.tests.test_source_task_memory import setup


def test_spectral_direction_splits_hidden_shift_without_changing_weight():
    torch.manual_seed(7)
    weight = torch.randn(6, 9)
    original = weight.clone()
    delta = torch.randn(11, 9)
    first, rest, info = split_first_direction(weight, delta)
    torch.testing.assert_close(first + rest, delta)
    h1, hr = first @ weight.T, rest @ weight.T
    torch.testing.assert_close((h1 * hr).sum(-1), torch.zeros(11), atol=2e-5, rtol=0)
    expected = torch.linalg.svdvals(weight.double()).square()
    assert info['weight_first_energy_fraction'] == pytest.approx(float(expected[0]/expected.sum()))
    torch.testing.assert_close(weight, original, atol=0, rtol=0)


def test_zero_readout_is_reported_not_assigned_a_fake_direction():
    with pytest.raises(ValueError, match='zero'):
        split_first_direction(torch.zeros(4, 9), torch.randn(2, 9))
    with pytest.raises(ValueError, match='finite'):
        split_first_direction(torch.full((4, 9), float('nan')), torch.randn(2, 9))


def test_rank_one_only_model_has_identical_full_and_first_risk():
    old, e = setup()
    model = ContextAdapter(old.base, forecast_weights=torch.ones(4, 23)).eval()
    with torch.no_grad():
        model.memory.output.weight.copy_(torch.randn(model.base.cfg.dim, 1) @ torch.randn(1, 92) * .02)
    e.update(labels=torch.ones(4), local_mask=torch.ones(4, dtype=torch.bool))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = evaluate_geometry(model, e, .7)
    assert result['geometry']['hidden_first_energy_fraction'] == pytest.approx(1., abs=1e-5)
    # Random test model has a tiny fast effect: dividing float32 roundoff by
    # that effect is ill-conditioned. Check actual logit error, not its ratio.
    logit_error = (result['geometry']['first_vs_full_logit_rms_ratio']
                   * result['geometry']['full_logit_change_rms'])
    assert logit_error < 1e-6
    assert result['modes']['first']['bag'] == pytest.approx(result['modes']['full']['bag'], abs=1e-6)
    assert result['modes']['rest']['bag'] == pytest.approx(result['modes']['off']['bag'], abs=1e-6)
    assert all(p.grad is None for p in model.parameters())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)


def test_weight_rank_does_not_imply_actual_hidden_shift_rank():
    weight = torch.diag(torch.tensor([100., 1.]))
    delta = torch.tensor([[0., 1.], [0., 2.]])
    _, _, info = split_first_direction(weight, delta)
    assert info['weight_first_energy_fraction'] > .999
    assert info['hidden_first_energy_fraction'] == 0.
