import numpy as np
import pytest
import torch


def test_oracle_requires_signal_descent_and_can_mix_conflicting_channels():
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.learning import solve_weights
    # Channel0 improves risk but worsens signal; channel1 compensates signal.
    w, r = solve_weights(np.array([[2., -1.]]), np.array([[-1., 2.]]))
    np.testing.assert_allclose(w, [2/3, 1/3], atol=1e-8)
    assert r['margin'] == pytest.approx(1.)
    w, r = solve_weights(np.array([[-2., -1.]]), np.ones((1,2)))
    assert r['margin'] < 0


def test_shared_maximin_does_not_sacrifice_one_history_for_mean_gain():
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.learning import solve_weights
    a = np.array([[2., -1.], [-1., 2.]])
    w, r = solve_weights(a, np.ones_like(a))
    np.testing.assert_allclose(w, [.5,.5], atol=1e-8)
    assert r['margin'] == pytest.approx(.5)
    with pytest.raises(ValueError):
        solve_weights(a, np.full_like(a, np.nan))


def test_feedback_clock_is_strict_and_future_labels_are_not_in_signal():
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.learning import feedback_prefix
    x = dict(values=np.arange(330*23,dtype='float32').reshape(330,23), available=np.ones((330,23),bool))
    p, count = feedback_prefix(x, 300)
    assert count == 276 and len(p['values']) == 276
    assert np.array_equal(p['values'][-1], x['values'][275])
    with pytest.raises(ValueError):
        feedback_prefix(dict(x,label=1), 300)
    with pytest.raises(ValueError):
        feedback_prefix(x, 250)


def test_channel_gradients_reconstruct_total_and_geometry_sign():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.learning import collect_geometry
    torch.manual_seed(51); model = small_model()
    x = dict(values=np.random.default_rng(8).normal(size=(330,23)).astype('float32'), available=np.ones((330,23),bool))
    record, state = collect_geometry(model,x,300,1.,device='cpu')
    assert record['last_observation'] == 275 and record['query'] == 300
    assert record['gradient_sum_relative_error'] < 1e-5
    g = state['parts'].sum(0); r=state['risk_gradient']
    np.testing.assert_allclose(sum(record['signal_alignment']), 1., atol=1e-5)
    expected = ((state['parts'].double()@r.double()) / (g.double().norm()*r.double().norm())).numpy()
    np.testing.assert_allclose(record['risk_alignment'],expected,atol=1e-5,rtol=1e-5)


def test_future_signal_cannot_change_arrived_gradients_and_zero_step_is_no_update():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.learning import collect_geometry, finite_step
    model = small_model()
    x = dict(values=np.random.default_rng(9).normal(size=(330,23)).astype('float32'), available=np.ones((330,23),bool))
    before = {k:v.clone() for k,v in model.state_dict().items()}
    record, state = collect_geometry(model,x,300,1.,device='cpu')
    modified = {k:v.copy() for k,v in x.items()}; modified['values'][276:] = 77.
    _, second = collect_geometry(model,modified,300,0.,device='cpu')
    torch.testing.assert_close(state['parts'],second['parts'],rtol=0,atol=0)
    no_update = finite_step(model,state,torch.zeros_like(state['risk_gradient']),.01)
    assert no_update['zero_direction'] and no_update['actual_step_norm'] == 0
    assert no_update['risk_after'] == record['risk_before']
    assert no_update['signal_after'] == record['signal_before']
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_query_selection_excludes_late_alarms_and_deduplicates_safe_peak():
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.run import select_queries
    times = np.arange(700)/1000
    scores = np.zeros(700); scores[555] = .9
    danger = select_queries(times,True,.699,scores)
    assert len(danger)==8 and min(danger)==574 and max(danger)==668
    safe = select_queries(times,False,None,scores)
    assert len(safe)==9 and 555 in safe and min(safe)==280 and max(safe)==699
