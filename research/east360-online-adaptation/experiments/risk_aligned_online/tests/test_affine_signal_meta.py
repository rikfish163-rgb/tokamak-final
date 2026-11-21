from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import affine_signal_meta as meta
from experiments.risk_aligned_online.feature_affine import FeatureAffineModel
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def case():
    torch.manual_seed(20261009)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.linspace(-1, 1, 23), torch.linspace(.5, 2, 23)).eval()
    model = FeatureAffineModel(base, forecast_coordinates='observed')
    e = {k: torch.randn(2, n, 23) for k, n in [('x', 128), ('qx', 128), ('truth', 25)]}
    e.update(a=torch.ones_like(e['x'], dtype=torch.bool),
             qa=torch.ones_like(e['qx'], dtype=torch.bool),
             mask=torch.ones_like(e['truth'], dtype=torch.bool), y=torch.tensor([0., 1.]))
    return model, e


def test_weights_are_positive_bounded_and_uniform_at_zero():
    logits = torch.zeros(23, requires_grad=True)
    weights = meta.channel_weights(logits)
    torch.testing.assert_close(weights, torch.ones(23), rtol=0, atol=0)
    weights = meta.channel_weights(torch.linspace(-1e5, 1e5, 23))
    assert weights.min() >= .1 and weights.max() <= 20.8
    torch.testing.assert_close(weights.mean(), torch.tensor(1.))


def test_uniform_weighted_loss_is_exact_archived_loss():
    model, e = case()
    prediction = model(e['x'], e['a'])['forecast']
    measured = meta.weighted_signal_loss(model, prediction, e['truth'], e['mask'], torch.ones(23))
    torch.testing.assert_close(measured.mean(), model.forecast_loss(prediction, e['truth'], e['mask']))
    # Actual streaming uses a batch of one; uniform weighting must be bitwise identical.
    measured = meta.weighted_signal_loss(model, prediction[:1], e['truth'][:1], e['mask'][:1], torch.ones(23))
    torch.testing.assert_close(measured.mean(), model.forecast_loss(prediction[:1], e['truth'][:1], e['mask'][:1]), rtol=0, atol=0)


def test_weighted_loss_uses_observed_weight_mass_and_never_gradients_truth():
    model, e = case()
    truth = e['truth'].clone().requires_grad_()
    mask = torch.zeros_like(e['mask']); mask[:, :, 0] = True
    prediction = torch.randn_like(truth, requires_grad=True)
    weights = torch.ones(23); weights[0] = 10.
    result = meta.weighted_signal_loss(model, prediction, truth, mask, weights)
    reference = meta.weighted_signal_loss(model, prediction, truth, mask, torch.ones(23))
    torch.testing.assert_close(result, reference)
    gradient, = torch.autograd.grad(result.sum(), prediction)
    assert not gradient[:, :, 1:].any()
    assert truth.grad is None


@pytest.mark.parametrize('weights', [torch.ones(22), torch.zeros(23), torch.full((23,), float('nan'))])
def test_invalid_signal_weights_are_rejected(weights):
    model, e = case()
    with pytest.raises(ValueError, match='weights'):
        meta.weighted_signal_loss(model, e['truth'], e['truth'], e['mask'], weights)


def test_synthetic_coordinate_shift_is_shared_by_past_future_and_query():
    model, e = case()
    e['qx'] = e['x'].clone(); e['truth'] = e['x'][:, :25].clone()
    before = {k: v.clone() for k, v in e.items()}
    moved, audit = meta.distort_episode(model, e, np.random.default_rng(8))
    torch.testing.assert_close(moved['qx'], moved['x'], rtol=0, atol=0)
    torch.testing.assert_close(moved['truth'], moved['x'][:, :25], rtol=0, atol=0)
    torch.testing.assert_close(moved['y'], e['y'], rtol=0, atol=0)
    assert len(audit['gain']) == 2 and len(audit['gain'][0]) == 23
    for k, v in before.items(): torch.testing.assert_close(e[k], v, rtol=0, atol=0)


def test_source_training_changes_only_loss_logits_and_has_real_meta_gradient():
    model, e = case()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    logits = torch.nn.Parameter(torch.zeros(23))
    optimizer = torch.optim.Adam([logits], lr=.03)
    record = meta.train_step(model, logits, optimizer, e)
    assert record['meta_gradient_norm'] > 0 and torch.isfinite(logits).all()
    assert bool(logits.any())
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    assert all(p.grad is None for p in model.base.parameters())
    assert model.affine.grad is None


def test_online_model_weights_only_intra_loss_not_persistent_loss_or_risk():
    model, e = case()
    weights = torch.ones(23); weights[0] = 8.
    weighted = meta.WeightedFeatureAffineModel(deepcopy(model.base), weights)
    fast = model.initial_fast(2).requires_grad_()
    original = model(e['x'], e['a'], fast)
    out = weighted(e['x'], e['a'], fast)
    for k in out: torch.testing.assert_close(out[k], original[k], rtol=0, atol=0)
    torch.testing.assert_close(weighted.forecast_loss(out['forecast'], e['truth'], e['mask']),
                              model.forecast_loss(out['forecast'], e['truth'], e['mask']), rtol=0, atol=0)
    assert not torch.equal(weighted.intra_forecast_loss(out['forecast'], e['truth'], e['mask']),
                           weighted.forecast_loss(out['forecast'], e['truth'], e['mask']))
    assert sum(p.numel() for p in weighted.persistent_parameters()) == 46


def test_weight_artifact_checks_source_parent_and_format(tmp_path):
    path = tmp_path / 'weights.pt'
    protocol = dict(format=meta.FORMAT, parent_sha256='a', source_only=True,
                    original_model_bitwise_unchanged=True)
    torch.save(dict(weights=torch.ones(23), protocol=protocol), path)
    weights, loaded = meta.load_weights(path, expected_parent_sha256='a')
    torch.testing.assert_close(weights, torch.ones(23))
    assert loaded == protocol
    with pytest.raises(ValueError, match='parent'):
        meta.load_weights(path, expected_parent_sha256='b')


def test_intra_runner_uses_explicit_weighted_method(monkeypatch):
    from experiments.risk_aligned_online.redesign_20261004 import intra
    model, e = case()
    calls = []
    def weighted(prediction, truth, available):
        calls.append(True)
        return prediction.square().mean() + 7.
    model.intra_forecast_loss = weighted
    value, _ = intra._loss(model, e['truth'], e['truth'], e['mask'], 'level',
                           torch.zeros(2, 23), torch.ones(2, 23, dtype=torch.bool))
    assert calls == [True] and value > 7.


@pytest.mark.parametrize('extra,expected', [([], 'reversible_affine'),
    (['--internal-site', 'reversible_affine', '--objective', 'forward_holdout'], 'level')])
def test_runner_rejects_weights_with_unmatched_training_contract(tmp_path, monkeypatch, capsys, extra, expected):
    import sys
    from experiments.risk_aligned_online import density_online
    monkeypatch.setattr(sys, 'argv', ['density_online', '--output', str(tmp_path / 'run'),
        '--source-signal-weights', str(tmp_path / 'weights.pt'), *extra])
    with pytest.raises(SystemExit): density_online.main()
    # Match the actual validation error, not a word also printed in usage.
    error_line = capsys.readouterr().err.strip().splitlines()[-1]
    assert 'source signal weights require reversible_affine with level objective' in error_line
