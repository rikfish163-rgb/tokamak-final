import numpy as np
import pytest
import torch

from .macro_credit import macro_plan, choose_pair, credit_examples, fit_rule, class_errors
from .macro_experiment import rescue_gate, global_pair, statistics


def signals(n=700):
    return dict(device='EAST', values=np.zeros((n, 23), np.float32),
                available=np.ones((n, 23), bool), times=np.arange(n)*.001)


def test_macro_clock_and_bounded_updates():
    q, decisions, events = macro_plan(signals(), torch.zeros(23), torch.ones(23), pair=(1, -1))
    assert [d['end'] for d in decisions] == [5, 265]
    assert not q[:6].any()
    assert q[6] == pytest.approx(.02)
    assert q[260] == pytest.approx(1.)
    assert q[265] == pytest.approx(1.)
    assert q[266] == pytest.approx(.98)
    assert q[520] == pytest.approx(0., abs=1e-14)
    assert np.allclose(q[520:], 0., atol=1e-14) and len(events) == 40


@pytest.mark.parametrize('arm', ['H', 'P'])
def test_prefix_unchanged_by_future_or_outcome(arm):
    s = signals()
    s['values'][300:] = 999
    rule = torch.zeros(3, 117, dtype=torch.float64)
    rule[1, -1] = -.01
    q, _, _ = macro_plan(s, torch.zeros(23), torch.ones(23), rule=rule, arm=arm)
    prefix = {k: v[:267] if isinstance(v, np.ndarray) else v for k, v in s.items()}
    prefix['is_disrupt'] = True  # Extra label must never enter the decision.
    qp, _, _ = macro_plan(prefix, torch.zeros(23), torch.ones(23), rule=rule, arm=arm)
    np.testing.assert_array_equal(qp, q[:267])


def test_missing_events_do_not_advance_macro_or_q():
    s = signals(80)
    s['available'][:27] = False
    q, decisions, events = macro_plan(s, torch.zeros(23), torch.ones(23), pair=(1, 1))
    assert decisions[0]['end'] == 31
    assert not q[:32].any() and q[32] == pytest.approx(.02)
    assert len(events) == 4 and len(decisions) == 1


def test_short_shot_does_not_have_future_decision():
    q, decisions, events = macro_plan(signals(5), torch.zeros(23), torch.ones(23), pair=(1, 1))
    assert not q.any() and decisions == events == []


def test_second_decision_context_not_dependent_on_second_action():
    a = macro_plan(signals(), torch.zeros(23), torch.ones(23), pair=(-1, 0))[1]
    b = macro_plan(signals(), torch.zeros(23), torch.ones(23), pair=(-1, 1))[1]
    assert a[1]['context'] == b[1]['context']
    assert a[1]['context'][-2] == pytest.approx(-1.)


def test_p_freezes_only_values_and_live_own_state():
    s = signals()
    s['values'][200:] = 3
    rule = torch.zeros(3, 117, dtype=torch.float64)
    rule[2, -1] = -1
    _, h, _ = macro_plan(s, torch.zeros(23), torch.ones(23), rule=rule)
    _, p, _ = macro_plan(s, torch.zeros(23), torch.ones(23), rule=rule, arm='P')
    assert h[1]['context'][:69] != h[0]['context'][:69]
    assert p[1]['context'][:69] == p[0]['context'][:69]
    assert p[1]['context'][69:] == h[1]['context'][69:]


def table(costs, label=0):
    c = np.zeros(117); c[-1] = 1
    second = []
    for q in (0., -1., 1.):
        cc = c.copy(); cc[-2] = q; second.append(cc.tolist())
    return dict(context_1=c.tolist(), context_2=second, costs=costs, disruptive=label)


def test_teacher_uses_student_continuation_not_oracle():
    t = table([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    zero = torch.zeros(3, 117, dtype=torch.float64)
    assert choose_pair(t, zero) == (0, 0)
    examples = credit_examples(t, zero)
    np.testing.assert_allclose(examples[0][1], np.array([1, 1, 0])-2/3)
    np.testing.assert_allclose(examples[1][1], np.array([1, 0, 1])-2/3)


def test_all_tied_costs_do_not_create_fictional_hold_labels():
    zero = torch.zeros(3, 117, dtype=torch.float64)
    data = [table(np.ones((3, 3)).tolist(), k) for k in (0, 1)]
    fitted, logs = fit_rule(data, zero)
    assert torch.equal(fitted, zero) and len(logs) == 3
    assert all(r['non_tied_examples'] == 0 for r in logs)


def test_cost_fit_learns_two_classes_without_risk_proxy():
    a = table([[1]*3, [0]*3, [1]*3], 0)
    b = table([[1]*3, [1]*3, [0]*3], 1)
    b['context_1'][0] = 1.
    for c in b['context_2']: c[0] = 1.
    fitted, _ = fit_rule([a,b], torch.zeros(3, 117, dtype=torch.float64))
    assert choose_pair(a, fitted)[0] == 1
    assert choose_pair(b, fitted)[0] == 2


def test_capacity_requires_real_baseline_errors_in_both_classes():
    a = table([[1]*3, [0]*3, [1]*3], 0); a['shot'] = 1
    b = table([[1]*3, [1]*3, [0]*3], 1); b['shot'] = 2
    assert rescue_gate([a,b])['passed']
    a['costs'][0][0] = 0
    assert not rescue_gate([a,b])['passed']


def test_global_schedule_is_class_balanced_and_hold_wins_exact_tie():
    a = table([[1,0,0],[1,1,1],[1,1,1]],0)
    b = table([[0,1,1],[0,0,0],[0,0,0]],1)
    assert global_pair([a,a,a,b]) == (0,0)


def test_tardy_is_not_timely_and_missing_class_has_no_rate():
    rows = [dict(disruptive=True,prediction=dict(alarm=True,timely=False))]
    out = statistics(rows)
    assert out['tp'] == 0 and out['tpr'] == 0 and out['fpr'] is None


def test_joint_trajectory_not_old_continuation_decides_acceptance(monkeypatch):
    # A constructed coordination failure: first hold was safe only when the
    # old rule's second action was negative. Simultaneous replacement loses it.
    a = table([[1]*3, [0]*3, [1]*3], 0)
    b = table([[1,0,1], [0,0,1], [1,1,1]], 1)
    initial = torch.zeros(3,117,dtype=torch.float64)
    negative = initial.clone(); negative[1,-1] = -1
    proposals = iter((negative.T, initial.T))
    monkeypatch.setattr(torch.linalg, 'solve', lambda *args: next(proposals))
    fitted, logs = fit_rule([a,b], initial, conservative=True)
    assert torch.equal(fitted, negative)
    assert class_errors([a,b], fitted) == [0,0]
    assert [r['accepted'] for r in logs] == [True,False]
    assert logs[1]['proposal_class_errors'] == [1,1]


def test_equal_cost_proposal_keeps_exact_current_rule(monkeypatch):
    data = [table(np.zeros((3,3)).tolist(), k) for k in (0,1)]
    zero = torch.zeros(3,117,dtype=torch.float64)
    changed = zero.clone(); changed[1,-1] = -1
    monkeypatch.setattr(torch.linalg, 'solve', lambda *args: changed.T)
    fitted, logs = fit_rule(data, zero, conservative=True)
    assert torch.equal(fitted, zero) and len(logs) == 1
    assert logs[0]['accepted'] is False


def test_conservative_rule_never_substitutes_one_class_for_other(monkeypatch):
    a = table([[0]*3,[1]*3,[1]*3],0)
    b = table([[1]*3,[0]*3,[1]*3],1)
    zero = torch.zeros(3,117,dtype=torch.float64)
    negative = zero.clone(); negative[1,-1] = -1
    monkeypatch.setattr(torch.linalg, 'solve', lambda *args: negative.T)
    fitted, logs = fit_rule([a,b], zero, conservative=True)
    assert torch.equal(fitted, zero)
    assert logs[0]['current_class_errors'] == [0,1]
    assert logs[0]['proposal_class_errors'] == [1,0]
