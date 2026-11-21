"""Source event/forecast redesign: causality, objectives, temporary update."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_adapt_20261022.model import (
    Config, EventForecastModel, arrived_loss, update_fast, event_loss,
)
from experiments.risk_aligned_online.event_adapt_20261022.data import (
    attach_training_events, fit_coordinates, feedback_sample,
)
from experiments.risk_aligned_online.event_prefix_probe import Event


@pytest.fixture
def model():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    return EventForecastModel(Config(dim=16, rank=4, horizon=8, dilations=(1, 2)),
                              np.zeros(23), np.ones(23), np.ones(23), np.ones(23, bool))


def signal(n=40):
    rng = np.random.default_rng(2)
    return dict(device='J-TEXT', shot=7, values=rng.normal(size=(n, 23)).astype('float32'),
                available=np.ones((n, 23), bool), times=.15+np.arange(n)/1000,
                y=np.zeros(n, 'float32'), valid=np.ones(n, bool), is_disrupt=False, down=None)


def test_forecast_and_risk_are_causal(model):
    x = torch.randn(2, 25, 23); m = torch.ones_like(x, dtype=torch.bool)
    a = model.sequence(x, m)
    x[:, 14:] += 50
    b = model.sequence(x, m)
    for key in ('logit', 'forecast', 'event'):
        torch.testing.assert_close(a[key][:, :14], b[key][:, :14])


def test_full_sequence_matches_rolling_window(model):
    x = torch.randn(2, 40, 23); m = torch.rand_like(x) > .1
    seq = model.sequence(x, m)
    for t in (0, 1, 15, 39):
        lo = max(0, t-model.cfg.window+1)
        y = model(x[:, lo:t+1], m[:, lo:t+1])
        torch.testing.assert_close(y['logit'], seq['logit'][:, t], atol=1e-6, rtol=1e-5)


def test_missing_inputs_do_not_change_outputs(model):
    x = torch.randn(2, 16, 23); m = torch.rand_like(x) > .3
    a = model(x, m)
    x[~m] = float('nan')
    b = model(x, m)
    for key in a:
        torch.testing.assert_close(a[key], b[key])


def test_risk_really_reads_predicted_evolution(model):
    x = torch.randn(2, 16, 23); m = torch.ones_like(x, dtype=torch.bool)
    out = model(x, m)
    grads = torch.autograd.grad(out['logit'].sum(), model.forecast_head.parameters())
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_unsupervised_forecast_slots_cannot_carry_hidden_risk_information(model):
    model.forecast_active[10] = False
    x = torch.randn(2, 16, 23); m = torch.ones_like(x, dtype=torch.bool)
    m[:, :, 2] = False
    before = model(x, m)['logit'].detach()
    with torch.no_grad():
        bias = model.forecast_head[-1].bias.reshape(model.cfg.horizon, 23)
        bias[:, 10] += 10000
        bias[:, 2] += 10000
    torch.testing.assert_close(before, model(x, m)['logit'])


def test_detached_route_control_has_no_forecast_to_risk_gradient():
    cfg = Config(dim=16, rank=4, horizon=8, dilations=(1, 2), risk_reads_forecast=False)
    model = EventForecastModel(cfg, np.zeros(23), np.ones(23), np.ones(23), np.ones(23, bool))
    x = torch.randn(2, 16, 23); m = torch.ones_like(x, dtype=torch.bool)
    out = model(x, m)
    grad = torch.autograd.grad(out['logit'].sum(), tuple(model.forecast_head.parameters()), allow_unused=True)
    assert all(g is None or not g.any() for g in grad)


def test_unknown_event_is_not_normal_and_does_not_enter_ce():
    logits = torch.randn(4, 6, requires_grad=True)
    loss = event_loss(logits, torch.tensor([-1, 2, -1, 3]))
    loss.backward()
    assert not logits.grad[[0, 2]].any() and logits.grad[[1, 3]].abs().sum() > 0
    assert event_loss(logits, torch.full((4,), -1)).item() == 0


def test_events_are_train_only_and_do_not_modify_risk_labels():
    s = signal()
    event = Event(7, 1, 2, .160, .175)
    result = attach_training_events([s], {7: [event]}, allowed_train_ids=[7])
    assert result[0]['event_target'][5] == -1
    assert result[0]['event_target'][12] == 2
    np.testing.assert_array_equal(result[0]['y'], s['y'])
    with pytest.raises(ValueError, match='TRAIN'):
        attach_training_events([s], {7: [event]}, allowed_train_ids=[9])


def test_forecast_supervision_waits_for_real_arrival():
    s = signal(); cfg = Config(dim=16, horizon=8, dilations=(1, 2))
    sample = feedback_sample(s, origin=15, now=18, cfg=cfg)
    assert sample['target_mask'][:3].all() and not sample['target_mask'][3:].any()
    assert np.all(sample['target'][3:] == 0)
    altered = dict(s, values=s['values'].copy())
    altered['values'][19:] = 999999
    again = feedback_sample(altered, origin=15, now=18, cfg=cfg)
    for k in ('x', 'mask', 'target', 'target_mask'):
        np.testing.assert_array_equal(sample[k], again[k])


def test_update_has_gradient_changes_future_risk_not_source(model):
    x = torch.randn(2, 16, 23); m = torch.ones_like(x, dtype=torch.bool)
    target = torch.randn(2, 8, 23); mask = torch.ones_like(target, dtype=torch.bool)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fast = model.initial_fast(2).detach().requires_grad_(True)
    old = model(x, m, fast)['logit'].detach()
    nxt, info = update_fast(model, fast, x, m, target, mask, lr=.1, radius=1.)
    assert info['updated'] and not torch.equal(fast, nxt)
    assert (model(x, m, nxt)['logit']-old).abs().max() > 0
    for k, v in before.items(): torch.testing.assert_close(v, model.state_dict()[k])
    torch.testing.assert_close(model(x, m)['logit'], old)


def test_no_matured_observation_no_update(model):
    x = torch.randn(2, 16, 23); m = torch.ones_like(x, dtype=torch.bool)
    fast = model.initial_fast(2).detach().requires_grad_(True)
    nxt, info = update_fast(model, fast, x, m, torch.zeros(2, 8, 23),
                            torch.zeros(2, 8, 23, dtype=torch.bool), lr=.1, radius=1.)
    assert not info['updated']
    torch.testing.assert_close(fast, nxt)


def test_meta_can_differentiate_through_same_inner_operator(model):
    x = torch.randn(2, 16, 23); m = torch.ones_like(x, dtype=torch.bool)
    fast = model.initial_fast(2)
    nxt, _ = update_fast(model, fast, x, m, torch.randn(2, 8, 23),
                        torch.ones(2, 8, 23, dtype=torch.bool), lr=.1, radius=1., create_graph=True)
    model(x+.1, m, nxt)['logit'].square().mean().backward()
    assert model.forecast_head[-1].weight.grad is not None
    assert torch.isfinite(model.forecast_head[-1].weight.grad).all()


def test_coordinates_reject_target_data_and_constant_channels():
    s = signal(); s['values'][:, 2] = 0
    c = fit_coordinates([s], allowed_train_ids=[7], horizon=8)
    assert np.isfinite(c['scale']).all() and (c['scale'] > 0).all()
    assert not c['forecast_active'][2]
    with pytest.raises(ValueError, match='TRAIN'):
        fit_coordinates([dict(s, device='EAST')], allowed_train_ids=[7], horizon=8)


def test_shuffled_event_control_keeps_input_batches_identical():
    from experiments.risk_aligned_online.event_adapt_20261022.train import event_pool, sample_batch
    from experiments.risk_aligned_online.simple_train import pools
    shots = [dict(signal(), shot=i, is_disrupt=bool(i % 2)) for i in range(1, 9)]
    for s in shots: s['y'][25:] = int(s['is_disrupt'])
    grouped = {s['shot']: [Event(s['shot'], 1, s['shot'] % 6, .155, .180)] for s in shots}
    real = attach_training_events(shots, grouped, allowed_train_ids=list(grouped))
    shuffle = attach_training_events(shots, grouped, allowed_train_ids=list(grouped), shuffle_seed=9)
    cfg = Config(dim=16, horizon=8, dilations=(1, 2))
    a = sample_batch(real, pools(real), event_pool(real), np.random.default_rng(17), cfg, 32, 'cpu')
    b = sample_batch(shuffle, pools(shuffle), event_pool(shuffle), np.random.default_rng(17), cfg, 32, 'cpu')
    for i in (0, 1, 2, 4, 5): torch.testing.assert_close(a[i], b[i])


def test_event_objective_reaches_encoder(model):
    x = torch.randn(2, 16, 23); m = torch.ones_like(x, dtype=torch.bool)
    loss = event_loss(model(x, m)['event'], torch.tensor([1, 4]))
    loss.backward()
    assert model.embedding.weight.grad.abs().sum() > 0


def test_inner_step_is_independent_of_other_shots_in_batch(model):
    x = torch.randn(2, 16, 23); m = torch.ones_like(x, dtype=torch.bool)
    truth = torch.randn(2, 8, 23); tm = torch.ones_like(truth, dtype=torch.bool)
    a, _ = update_fast(model, model.initial_fast(2), x, m, truth, tm, lr=.1, radius=1.)
    b, _ = update_fast(model, model.initial_fast(1), x[:1], m[:1], truth[:1], tm[:1], lr=.1, radius=1.)
    torch.testing.assert_close(a[:1], b)


def test_source_deadline_filter_keeps_labels_and_forecast_targets():
    from experiments.risk_aligned_online.event_adapt_20261022.train import deadline_eligible, risk_pools
    s = signal(); s['is_disrupt'] = True; s['down'] = .189; s['y'][15:] = 1
    original = s['y'].copy()
    keep = deadline_eligible(s, 10.)
    assert keep[28] and not keep[29:].any()
    groups = risk_pools([s], 10.)
    assert groups[1][0][1].max() == 28
    np.testing.assert_array_equal(s['y'], original)
    cfg = Config(dim=16, horizon=8, dilations=(1, 2))
    sample = feedback_sample(s, origin=28, now=36, cfg=cfg)
    assert sample['target_mask'].all()  # later signal targets are still legitimate
