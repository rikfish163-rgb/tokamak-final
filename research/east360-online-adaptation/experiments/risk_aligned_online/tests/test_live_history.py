import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.live_history_20261023.learning import (
    ARMS, current_geometry, backtrack, predict, HISTORY_TOL,
)
from experiments.risk_aligned_online.history_tangent_20261023.learning import flat, logit
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.tests.test_label_cone import setup_case


def test_batched_jacobian_matches_independent_loop_and_changes_with_fast_state():
    model, _, _, anchors = setup_case(); fast = initial_encoder(model)
    rows, z, losses = current_geometry(model, anchors, fast)
    other = logit(model, anchors['x'], anchors['mask'], fast)
    loop = torch.stack([flat(torch.autograd.grad(other[j], tuple(fast.values()), retain_graph=True))
                        for j in range(len(other))])
    torch.testing.assert_close(rows, loop, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(z, other.detach(), rtol=0, atol=0)
    expected = np.logaddexp(0., z.double().numpy())-anchors['labels'].numpy()*z.numpy()
    np.testing.assert_allclose(losses.numpy(), expected, rtol=1e-12, atol=1e-12)
    fast[next(iter(fast))] = (next(iter(fast.values()))+.01).detach().requires_grad_(True)
    changed, _, _ = current_geometry(model, anchors, fast)
    assert not torch.equal(changed, rows)


def test_backtracking_checks_real_loss_and_does_not_accumulate_tolerance():
    fast = {'x': torch.tensor([1.], requires_grad=True)}
    loss = lambda state: state['x'].detach().double().square()
    updated, receipt = backtrack(fast, torch.tensor([3.]), loss, loss(fast), loss(fast))
    assert receipt['fraction']==.5 and len(receipt['trials'])==2
    assert float(updated['x'].detach())==pytest.approx(-.5)
    assert receipt['trials'][0]['accepted'] is False
    # The current state can be better than the original; do not let it regress.
    improved = {'x': torch.tensor([.5], requires_grad=True)}
    frozen, receipt = backtrack(improved, torch.tensor([-1.]), loss,
                               loss(improved), torch.tensor([1.], dtype=torch.float64))
    assert receipt['fraction']==0 and receipt['reason']=='history_guard_rejected'
    assert frozen is improved and len(receipt['trials'])==6
    # Original loss remains a fixed ceiling, so the 1e-6 tolerance cannot accrue.
    current = {'x': torch.tensor([0.], requires_grad=True)}
    epsilon_loss = lambda state: torch.tensor([.9*HISTORY_TOL], dtype=torch.float64)+state['x'].detach().double().abs()
    updated, receipt = backtrack(current, torch.tensor([.5*HISTORY_TOL]), epsilon_loss,
                                epsilon_loss(current), torch.zeros(1, dtype=torch.float64))
    assert receipt['fraction']==.125
    assert float(epsilon_loss(updated))<=HISTORY_TOL


def test_backtracking_nonfinite_is_error_not_silently_accepted_smaller_step():
    fast = {'x': torch.tensor([1.], requires_grad=True)}
    with pytest.raises(FloatingPointError):
        backtrack(fast, torch.tensor([1.]), lambda state: torch.tensor([float('nan')]),
                  torch.ones(1), torch.ones(1))


def test_stream_causality_reset_and_guarded_norm_control():
    model, s, _, anchors = setup_case()
    parent = {k:v.clone() for k,v in model.state_dict().items()}
    curves, updates, drift = predict(model, s, anchors, device='cpu')
    again, uu, dd = predict(model, s, anchors, device='cpu')
    assert updates==uu and drift==dd
    s['values'][300:]=1000
    changed, _, _=predict(model, s, anchors, device='cpu')
    for a in ARMS:
        np.testing.assert_array_equal(curves[a], again[a])
        np.testing.assert_array_equal(curves[a][:300], changed[a][:300])
        assert [r['first_affected_prediction'] for r in updates[a]]==[256,384]
    for r, control in zip(updates['guarded'], updates['norm_only']):
        assert abs(r['parameter_delta_norm']-control['parameter_delta_norm'])<2e-7
        assert control['requested_step_norm']==r['accepted_step_norm']
        assert max(r['history_bce_change'])<=HISTORY_TOL
        assert max(r['history_change_from_parent'])<=HISTORY_TOL
    for k,v in model.state_dict().items():torch.testing.assert_close(v,parent[k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(model,dict(s,y=1),anchors,device='cpu')


def test_no_observed_targets_never_uses_history_to_train_instead():
    model,s,_,anchors=setup_case();s['available'][:]=False
    _,updates,_=predict(model,s,anchors,device='cpu')
    for a in ARMS:
        assert all(r['gradient_steps']==0 and r['parameter_delta_norm']==0 for r in updates[a])
