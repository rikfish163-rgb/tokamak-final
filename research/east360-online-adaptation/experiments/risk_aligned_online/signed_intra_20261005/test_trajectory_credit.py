import itertools

import numpy as np
import pytest
import torch

from .context import ACTIONS
from .test_policy import signals
from .trajectory_credit import (
    sampled_plan, first_alarm_index, trajectory_log_probability, paired_loss,
    load_audit_after_credit,
)


def event(action, effect, value=1.):
    context = np.zeros(117); context[-1] = value
    return dict(context=context.tolist(), action=action, effect=effect)


def test_random_plan_is_prefix_causal_and_resets():
    source = signals(130)
    rule = torch.zeros(3, 117, dtype=torch.float64)
    center, scale = torch.zeros(23), torch.ones(23)
    q, events = sampled_plan(source, rule, center, scale, np.random.default_rng(131))
    prefix = {k: v[:40] if isinstance(v, np.ndarray) else v for k, v in source.items()}
    qp, ep = sampled_plan(prefix, rule, center, scale, np.random.default_rng(131))
    np.testing.assert_array_equal(q[:40], qp)
    assert not q[:6].any() and ep[0]['q_before'] == 0
    assert [(e['end'], e['action']) for e in events if e['end'] < 40] == [
        (e['end'], e['action']) for e in ep]
    np.testing.assert_array_equal(q, sampled_plan(
        source, rule, center, scale, np.random.default_rng(131))[0])


def test_invalid_events_do_not_draw_or_update():
    source = signals(60); source['available'][:] = False
    rng = np.random.default_rng(3); before = rng.bit_generator.state
    q, events = sampled_plan(source, torch.zeros(3, 117, dtype=torch.float64),
                             torch.zeros(23), torch.ones(23), rng)
    assert not q.any() and not events and rng.bit_generator.state == before


def test_actor_rejects_outcome_fields():
    source = signals(); source['is_disrupt'] = True
    with pytest.raises(ValueError, match='signal-only'):
        sampled_plan(source, torch.zeros(3, 117), torch.zeros(23), torch.ones(23),
                     np.random.default_rng(0))


def test_absorbing_alarm_is_first_valid_crossing_not_peak():
    shot = dict(times=np.arange(6)*.001, valid=np.array([0,1,1,1,1,1], bool),
                is_disrupt=False, down=None)
    assert first_alarm_index(shot, np.array([.99,.2,.8,.1,.95,.9]), .7) == 2
    assert first_alarm_index(shot, np.array([.99,.2,.3,.1,.5,.6]), .7) is None
    shot.update(is_disrupt=True, down=.002)
    assert first_alarm_index(shot, np.array([.99,.2,.3,.1,.95,.9]), .7) is None


def test_credit_includes_action_at_alarm_but_not_after_or_no_effect():
    rule = torch.zeros(3, 117, dtype=torch.float64, requires_grad=True)
    events = [event(1, 6), event(-1, 26), event(1, 32), event(1, None)]
    value = trajectory_log_probability(rule, events, 26)
    torch.testing.assert_close(value, torch.tensor(-2*np.log(3), dtype=rule.dtype))
    grad = torch.autograd.grad(value, rule)[0]
    assert grad[ACTIONS.index(-1), -1] > 0
    assert grad[ACTIONS.index(1), -1] > 0
    torch.testing.assert_close(trajectory_log_probability(rule, events, None),
                               torch.tensor(-3*np.log(3), dtype=rule.dtype))
    torch.testing.assert_close(trajectory_log_probability(rule, events, 2), rule.sum()*0)


def test_same_cost_has_exact_zero_gradient_and_no_arbitrary_hold_target():
    rule = torch.randn(3, 117, dtype=torch.float64, requires_grad=True)
    a = dict(events=[event(1, 6)], cost=1, first_alarm=None)
    b = dict(events=[event(-1, 6)], cost=1, first_alarm=None)
    loss = paired_loss(rule, a, b)
    assert loss.item() == 0 and not torch.autograd.grad(loss, rule)[0].any()


def test_pairwise_estimator_matches_exact_tiny_policy_gradient():
    rule = torch.zeros(3, 117, dtype=torch.float64, requires_grad=True)
    with torch.no_grad(): rule[:, -1] = torch.tensor([.2,-.3,.7])
    probabilities = rule[:, -1].softmax(0)
    costs = [1, 1, 0]
    expected_cost = (probabilities * probabilities.new_tensor(costs)).sum()
    exact = torch.autograd.grad(expected_cost, rule, retain_graph=True)[0]
    estimate = rule.sum()*0
    for i,j in itertools.product(range(3), repeat=2):
        left = dict(events=[event(ACTIONS[i], 6)], cost=costs[i], first_alarm=None)
        right = dict(events=[event(ACTIONS[j], 6)], cost=costs[j], first_alarm=None)
        estimate = estimate + (probabilities[i]*probabilities[j]).detach()*paired_loss(rule,left,right)
    torch.testing.assert_close(torch.autograd.grad(estimate, rule)[0], exact)


def test_bad_nonbinary_cost_rejected():
    rule = torch.zeros(3,117)
    a = dict(events=[],cost=.25,first_alarm=None)
    with pytest.raises(ValueError, match='binary'):
        paired_loss(rule,a,dict(a,cost=1))


def test_absorbing_two_step_pair_gradient_matches_exact_full_enumeration():
    # Later action probabilities depend on the first sampled action. A first
    # alarm terminates credit but the full trajectory is still generated.
    rule = torch.zeros(3,117,dtype=torch.float64,requires_grad=True)
    with torch.no_grad(): rule[:,-1] = torch.tensor([.15,-.2,.4])
    trajectories=[]; probabilities=[]
    for a,b in itertools.product(ACTIONS,repeat=2):
        e1,e2 = event(a,6),event(b,26,1+.2*a)
        p1 = rule[:,-1].softmax(0)[ACTIONS.index(a)]
        p2 = (rule[:,-1]*(1+.2*a)).softmax(0)[ACTIONS.index(b)]
        alarm = 6 if a==1 else (26 if b==1 else None)
        trajectories.append(dict(events=[e1,e2],cost=int(alarm is not None),first_alarm=alarm))
        probabilities.append(p1*p2)
    expected_cost = sum(p*t['cost'] for p,t in zip(probabilities,trajectories))
    exact = torch.autograd.grad(expected_cost,rule,retain_graph=True)[0]
    estimate = rule.sum()*0
    for i,j in itertools.product(range(9),repeat=2):
        estimate = estimate + (probabilities[i]*probabilities[j]).detach()*paired_loss(
            rule,trajectories[i],trajectories[j])
    torch.testing.assert_close(torch.autograd.grad(estimate,rule)[0],exact)


@pytest.mark.parametrize('counts', [[0,0],[2,0],[0,3]])
def test_no_audit_waveform_loader_when_a_class_has_no_task_credit(counts):
    calls=[]
    assert load_audit_after_credit(counts,[1,2],lambda item:calls.append(item)) is None
    assert calls == []


def test_credit_gate_loads_only_fixed_audit_items_when_both_classes_contrast():
    calls=[]
    def loader(item):
        calls.append(item)
        return item*2
    assert load_audit_after_credit([1,2],[7,9],loader)==[14,18]
    assert calls==[7,9]
