import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.task_geometry_20261023.learning import (
    importance_weights, matched_direction, select_endpoints, target_loss,
)


def test_importance_downweights_sensitive_coordinates_and_shuffle_preserves_layers():
    f = {'a': torch.tensor([0., 2., 6.]), 'b': torch.tensor([1., 3.])}
    a, shuffled = importance_weights(f)
    assert a['a'][0] > a['a'][1] > a['a'][2]
    for name in a:
        torch.testing.assert_close(a[name].sort().values, shuffled[name].sort().values)
    again = importance_weights(f)[1]
    for k in f:
        torch.testing.assert_close(shuffled[k], again[k])
    with pytest.raises(ValueError):
        importance_weights({'a': torch.zeros(3)})


def test_direction_preserves_raw_clipped_norm_not_coordinate_direction():
    g = {'x': torch.tensor([3., 4.])}
    out, info = matched_direction(g, {'x': torch.tensor([.1, 1.])})
    assert torch.linalg.vector_norm(out['x']).item() == pytest.approx(1.)
    assert out['x'][1]/out['x'][0] > 10
    assert info['raw_norm'] == pytest.approx(5.)
    tiny, _ = matched_direction({'x': g['x']*.01}, {'x': torch.ones(2)})
    torch.testing.assert_close(tiny['x'], g['x']*.01)
    zero, info = matched_direction({'x': torch.zeros(2)}, {'x': torch.ones(2)})
    assert not zero['x'].any() and info['zero_direction']


def test_historical_endpoint_sampling_is_balanced_and_valid():
    s = dict(is_disrupt=True, valid=np.array([True]*40+[False]*3),
             y=np.array([0.]*30+[1.]*13))
    rows = select_endpoints(s)
    assert len(rows) == 32 and sum(s['y'][r] for r in rows) == 16
    assert max(rows) < 40
    s['is_disrupt'] = False
    assert len(select_endpoints(s)) == 32


def test_omitted_targets_change_loss_not_original_input_or_denominator():
    class Base:
        center = torch.zeros(23)
        scale = torch.ones(23)
    pred = torch.ones(1, 2, 23, requires_grad=True)
    truth = torch.zeros_like(pred); mask = torch.ones_like(pred, dtype=torch.bool)
    loss = target_loss(Base(), pred, truth, mask, omit=True)
    assert loss.item() == pytest.approx(.5*20/23)
    gradient, = torch.autograd.grad(loss, pred)
    for c in (0, 12, 13):
        assert not gradient[:, :, c].any()
    assert mask.all() and not truth.any()


def small_model():
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback
    torch.set_num_threads(2); torch.manual_seed(131)
    model = PairedFeedback(Simple23(Config(window=128, dim=16, heads=2, fast_site='tokens'),
                                   np.zeros(23), np.ones(23)), np.ones(23))
    model.set_stage('frozen')
    return model


def test_stream_causality_reset_parent_and_equal_preupdate_predictions():
    from experiments.risk_aligned_online.task_geometry_20261023.learning import predict, ARMS
    from experiments.risk_aligned_online.deepcoral_prefix_20261022.run import predict_pairs
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    model = small_model(); before = {k:v.clone() for k, v in model.state_dict().items()}
    signal = dict(values=np.random.default_rng(5).normal(size=(400,23)).astype('float32'),
                  available=np.ones((400,23),bool))
    weights = {a:{k:torch.ones_like(v) for k, v in initial_encoder(model).items()}
               for a in ('importance','shuffled')}
    predictions, updates = predict(model, signal, weights, device='cpu')
    old, _ = predict_pairs({'m':model}, signal, device='cpu')
    for arm in ARMS:
        np.testing.assert_array_equal(predictions[arm][:256], old['m_static'][:256])
        assert [u['first_affected_prediction'] for u in updates[arm]] == [256,384]
        assert all(u['parameter_delta_norm'] <= .00101 for u in updates[arm])
    np.testing.assert_allclose(predictions['importance'], old['m_intra'], rtol=1e-5, atol=1e-7)
    signal['values'][256:] = 999
    altered, receipts = predict(model, signal, weights, device='cpu')
    again, again_receipts = predict(model, signal, weights, device='cpu')
    for arm in ARMS:
        np.testing.assert_array_equal(predictions[arm][:256], altered[arm][:256])
        np.testing.assert_array_equal(altered[arm], again[arm])
        assert updates[arm][0] == receipts[arm][0]
    assert receipts == again_receipts
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    with pytest.raises(ValueError):
        predict(model, dict(signal,y=1), weights, device='cpu')


def test_importance_uses_single_endpoint_squares_with_declared_masses():
    from experiments.risk_aligned_online.task_geometry_20261023.learning import estimate_importance
    model = small_model()
    signal = dict(values=np.random.default_rng(2).normal(size=(160,23)).astype('float32'),
                  available=np.ones((160,23),bool), valid=np.ones(160,bool))
    shots = [dict(signal,shot=1,is_disrupt=False,y=np.zeros(160)),
             dict(signal,shot=2,is_disrupt=True,y=np.array([0.]*140+[1.]*20))]
    f, rows = estimate_importance(model, shots, device='cpu')
    assert len(rows) == 64 and sum(r['weight'] for r in rows) == pytest.approx(1.)
    for d in (False,True):
        assert sum(r['weight'] for r in rows if r['is_disrupt'] == d) == pytest.approx(.5)
    assert sum(v.sum().item() for v in f.values()) > 0
    assert all(torch.isfinite(v).all() and (v >= 0).all() for v in f.values())
    with pytest.raises(ValueError):
        estimate_importance(model, shots[:1], device='cpu')
