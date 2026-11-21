import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.ended_adaptation import fit_ended, three_group_batch
from experiments.risk_aligned_online.simple23 import Config, Simple23


def example():
    rng = np.random.default_rng(3)
    return dict(shot=7, device='EAST', is_disrupt=True, values=rng.normal(size=(60, 23)).astype('float32'),
                available=np.ones((60, 23), bool), valid=np.ones(60, bool),
                y=np.r_[np.zeros(30), np.ones(30)].astype('float32'))


def model():
    torch.manual_seed(7)
    return Simple23(Config(window=8, patch=4, dim=8, heads=2, layers=2),
                    np.zeros(23), np.ones(23))


def test_only_ended_ids_can_train():
    with pytest.raises(ValueError, match='ended'):
        fit_ended(model(), [example()], ended_ids=[8], steps=1, forecast_steps=1)


def test_scoped_supervision_then_forecast_does_not_change_classifier():
    net = model()
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    result = fit_ended(net, [example()], ended_ids=[7], steps=2, forecast_steps=2)
    assert len(result['supervised_loss']) == 2
    assert len(result['forecast_loss']) == 2
    assert result['forecast_stage_risk_max_delta'] == 0.
    assert result['changed_parameters']
    for name, param in net.named_parameters():
        if not name.startswith(('risk_head.', 'encoder.layers.1.', 'forecast_head.')) and name != 'fast_origin':
            torch.testing.assert_close(param, before[name], rtol=0, atol=0)
        assert param.requires_grad
    assert not torch.equal(net.forecast_head.weight, before['forecast_head.weight'])


def test_invalid_training_budget_fails():
    with pytest.raises(ValueError, match='positive'):
        fit_ended(model(), [example()], ended_ids=[7], steps=0, forecast_steps=1)


def test_three_groups_have_explicit_shares_and_never_sample_invalid_endpoints():
    disruptive = example()
    safe = example()
    safe.update(shot=8, is_disrupt=False, y=np.zeros(60, np.float32))
    safe['valid'][::2] = False
    safe['values'][:] = 99.
    x, mask, y, counts = three_group_batch([disruptive, safe], np.random.default_rng(4),
                                         64, model().cfg, 'cpu')
    assert counts == {'precursor': 32, 'safe_shot': 16, 'early_disruptive': 16}
    assert int(y.sum()) == 32
    assert int((x[:, -1, 0] == 99).sum()) == 16
    assert mask.dtype == torch.bool


def test_missing_safe_group_fails_instead_of_silently_changing_protocol():
    with pytest.raises(ValueError, match='safe_shot'):
        three_group_batch([example()], np.random.default_rng(4), 64, model().cfg, 'cpu')


def test_fit_logs_real_three_group_counts():
    safe = example()
    safe.update(shot=8, is_disrupt=False, y=np.zeros(60, np.float32))
    result = fit_ended(model(), [example(), safe], ended_ids=[7, 8], steps=2,
                       forecast_steps=1, sampling='three_group')
    assert result['sample_counts'] == {'precursor': 64, 'safe_shot': 32, 'early_disruptive': 32}


def test_relative_scope_preserves_all_original_classifier_weights():
    from experiments.risk_aligned_online.simple23 import add_relative_input
    net = add_relative_input(model())
    before = {n:p.clone() for n,p in net.named_parameters()}
    result = fit_ended(net, [example()], ended_ids=[7], steps=2, forecast_steps=1, update_scope='relative')
    assert 'relative_embedding.weight' in result['changed_parameters']
    for name,param in net.named_parameters():
        if not name.startswith(('relative_embedding.', 'forecast_head.')):
            torch.testing.assert_close(param,before[name],rtol=0,atol=0)
