import numpy as np
import pytest
import torch


def test_pair_loss_cancels_common_score_shift_and_has_opposite_gradients():
    from experiments.risk_aligned_online.within_shot_pair_20261024.learning import pair_loss
    early = torch.tensor([.3, -.5], requires_grad=True)
    late = torch.tensor([.7, .2], requires_grad=True)
    loss = pair_loss(late, early, 'rank').mean()
    ge, gl = torch.autograd.grad(loss, (early, late))
    assert (ge > 0).all() and (gl < 0).all()
    torch.testing.assert_close(ge, -gl)
    torch.testing.assert_close(pair_loss(late + 5, early + 5, 'rank'), pair_loss(late, early, 'rank'))
    expected = .5 * (torch.nn.functional.softplus(early) + torch.nn.functional.softplus(-late))
    torch.testing.assert_close(pair_loss(late, early, 'augmented'), expected)
    with pytest.raises(ValueError):
        pair_loss(late, early, 'invented')


def test_historical_partner_is_full_earlier_observation_not_boundary_or_future():
    from experiments.risk_aligned_online.within_shot_pair_20261024.learning import earlier_partner
    x = np.ones((600, 23), dtype='float32')
    x[:250] = 3
    x[550:] = 1e4
    signal = dict(values=x, available=np.ones_like(x, bool))
    end, distance, count = earlier_partner(signal, np.arange(600)/1000, .6, 500, np.zeros(23), np.ones(23))
    assert 127 <= end < 350 and count > 0 and np.isfinite(distance)
    x[550:] = -1e4
    assert (end, distance, count) == earlier_partner(signal, np.arange(600)/1000, .6, 500, np.zeros(23), np.ones(23))
    with pytest.raises(ValueError, match='earlier'):
        earlier_partner(signal, np.arange(600)/1000, .3, 250, np.zeros(23), np.ones(23))


def test_unpaired_control_preserves_marginals_but_changes_disruptive_shot():
    from experiments.risk_aligned_online.within_shot_pair_20261024.learning import permuted_partners
    shots = np.repeat([10, 20, 30, 40], 3)
    labels = np.repeat([1, 0, 1, 1], 3)
    ix = permuted_partners(shots, labels)
    np.testing.assert_array_equal(np.sort(ix), np.arange(12))
    assert (shots[ix[labels == 1]] != shots[labels == 1]).all()
    np.testing.assert_array_equal(ix[labels == 0], np.flatnonzero(labels == 0))


def test_qualification_cannot_be_claimed_from_equal_net_counts():
    from experiments.risk_aligned_online.within_shot_pair_20261024.run import eligible
    arms = ('off', 'point', 'augmented', 'paired', 'unpaired', 'reverse', 'uniform')
    rows = {a:dict(tp=132, fp=37, n_safe=180, n_disrupt=180) for a in arms}
    assert not eligible(rows)
    rows['paired'].update(tp=133, fp=35)
    assert eligible(rows)
    rows['unpaired'].update(tp=134, fp=36)
    assert not eligible(rows)


def test_new_runner_preserves_point_control_causality_and_temporary_state():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.trajectory_memory_20261024.learning import make_bank
    from experiments.risk_aligned_online.event_pairing_20261024.learning import Encoder
    from experiments.risk_aligned_online.event_routed_20261024.learning import prepare_bank, predict as old_predict
    from experiments.risk_aligned_online.within_shot_pair_20261024.learning import prepare_pairs, predict, ARMS
    model = small_model(); rng = np.random.default_rng(24)
    parent = {k:v.clone() for k,v in model.state_dict().items()}
    history = []
    for sid, disruptive in enumerate([True, False, True, False]):
        x = rng.normal(size=(600,23)).astype('float32')
        history.append(dict(shot=sid, values=x, available=np.ones_like(x,bool), times=np.arange(600)/1000,
            valid=np.ones(600,bool), is_disrupt=disruptive, down=.6 if disruptive else None))
    bank = make_bank(model, history, device='cpu')
    heads = {f'{mode}_{seed}':Encoder().state_dict() for mode in ('risk','mean','shuffle','family') for seed in (131,132,133)}
    prepare_bank(bank, heads, dict(mean=np.zeros(391),scale=np.ones(391)), device='cpu')
    _, pairs = prepare_pairs(model, bank, {s['shot']:s for s in history}, device='cpu')
    assert len(pairs) == 32 and all(r['early_lead_ms'] > 250 for r in pairs)
    for g in bank['pair_gradients'].values():
        torch.testing.assert_close(g[bank['labels']==0], bank['gradients'][bank['labels']==0], rtol=0, atol=0)
    x = rng.normal(size=(300,23)).astype('float32')
    s = dict(values=x, available=np.ones_like(x,bool))
    old, _ = old_predict(model, s, bank, device='cpu')
    curves, receipts = predict(model, s, bank, device='cpu')
    np.testing.assert_array_equal(curves['point'], old['risk'])
    np.testing.assert_array_equal(curves['off'], old['off'])
    repeated, again = predict(model, s, bank, device='cpu')
    assert receipts == again
    for arm in curves:
        np.testing.assert_array_equal(curves[arm], repeated[arm])
        np.testing.assert_array_equal(curves[arm][:128], curves['off'][:128])
    s['values'][256:] = 1e4
    changed, rc = predict(model, s, bank, device='cpu')
    assert receipts == rc
    for arm in curves:
        np.testing.assert_array_equal(curves[arm][:256], changed[arm][:256])
    for r in receipts:
        for arm in ARMS:
            assert abs(r[arm]['step_norm']-r['point']['step_norm']) < 2e-6
        for arm in ('augmented','paired','unpaired'):
            assert r[arm]['selected'] == r['point']['selected']
            assert r[arm]['weights'] == r['point']['weights']
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v, parent[k], atol=0, rtol=0)
