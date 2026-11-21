import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_feedback_dictionary_20261114 as trial
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_source_deadline_is_strict_and_a_completed_block_can_act_before_full_handoff():
    times = np.arange(201) / 1000
    q = np.array([127, 159])
    eligible, slots, ramp = trial.active_windows(q, np.array([True, True]), len(times), 8)
    assert not eligible[157] and not eligible[158]
    assert eligible[159] and eligible[189]
    assert not eligible[190] and eligible[191]
    assert slots[159] == 1 and slots[191] == 2
    assert ramp[159] == 1/8
    assert ramp[166] == 1
    mask = trial.teaching_window(times, np.ones(len(times), bool), np.ones(len(times), bool), .170, eligible)
    np.testing.assert_array_equal(np.flatnonzero(mask), [159])


def test_no_released_block_no_write_even_if_an_earlier_block_was_ready():
    q = np.array([127, 159])
    ok, slot, ramp = trial.active_windows(q, np.array([True, False]), 220, 8)
    assert ok[189] and not ok[190:].any()
    assert slot[190] == 2 and ramp[190] == 0.


def test_release_reference_is_anchored_to_the_issue_and_retains_sign():
    points = torch.zeros(1, 25, 2)
    points[:, 5:, 0] = 2.
    points[:, :, 1] = -2.
    actual = trial.departure(points, scale=torch.tensor([2., 1.]))
    torch.testing.assert_close(actual[0, 1, 0], torch.asinh(torch.tensor(1.)))
    torch.testing.assert_close(actual[0, :, 1], torch.asinh(torch.tensor(-2.)).expand(5))
    torch.testing.assert_close(actual[0, 0, 0], torch.tensor(0.))


def test_dictionary_does_not_encode_a_static_bias_and_can_separate_motion():
    groups = torch.eye(2)
    forces = torch.zeros(2, 5, 2)
    assert not trial.dictionary(forces, groups).any()
    f = torch.tensor([[[1., 1.], [-1., -1.], [0., 0.], [-1., -1.], [1., 1.]]])
    basis = trial.dictionary(f, groups)
    assert not basis[:, ::2].any()
    torch.testing.assert_close(basis[:, 1::2].sum(1), f)
    f[:, 0] = .5
    f[:, 1] = .5
    f[:, 2] = -.75
    f[:, 3:] = -.125
    basis = trial.dictionary(f, groups)
    torch.testing.assert_close(basis[:, ::2], torch.zeros_like(basis[:, ::2]))
    torch.testing.assert_close(basis[:, 1::2].sum(1), f)


def test_equal_dose_shadow_update_cannot_silently_modify_the_no_feedback_arm():
    coef = torch.tensor([[[[2., 0.],[0., 0.]], [[0., 2.],[0., 0.]]]])
    f = torch.tensor([.25])
    retained = trial.transport(coef[:, 1:], coef[:, :1], f, torch.tensor([True]))
    torch.testing.assert_close(retained, torch.tensor([[[[1.5, .5], [0., 0.]]]]))
    first = trial.transport(coef[:, 1:], coef[:, :1], f, torch.tensor([False]))
    torch.testing.assert_close(first, coef[:, 1:])


def test_source_fit_batches_use_the_same_episode_contract_as_the_collector():
    fields = (torch.tensor([[.2],[.3]]),)
    examples = [(fields, dict(sample_ends=[166,167],inactive_peak=-1.,positive=True)),
                ((fields[0][:1],),dict(sample_ends=[200],inactive_peak=-.5,positive=False))]
    rows, shot, inactive, disrupted, _ = trial.source_batches(examples, [1,0], 'cpu')
    torch.testing.assert_close(rows[0].flatten(), torch.tensor([.2,.2,.3]))
    torch.testing.assert_close(shot, torch.tensor([0,1,1]))
    torch.testing.assert_close(inactive, torch.tensor([-.5,-1.]))
    torch.testing.assert_close(disrupted, torch.tensor([False,True]))


def test_actual_model_update_can_train_a_policy_at_the_same_partial_step_as_inference():
    torch.manual_seed(55)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,layers=2,horizon=3),
                    torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(2,16,23)
    m = torch.ones_like(x,dtype=torch.bool)
    b = torch.randn(2,2,8,12)
    context = torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]])
    w = torch.zeros(2,4,8,requires_grad=True)
    ramp = torch.tensor([.25,1.])
    correction = lambda p: p.clone()*1.015  # noqa: E731
    _,full,_ = trial.group.forward_hidden(base,x,m,correction,writes=torch.ones(1,2,3,4))
    _,dose,err = trial.group.forward_hidden(base,x,m,correction,writes=torch.ones(2,2,3,4),
                                           radius_fraction=ramp)
    torch.testing.assert_close(dose,full*ramp,atol=1e-6,rtol=0.)
    assert err<1e-6
    rows = (None,)*6+(b,b,context,context,ramp,torch.tensor([False,True]))
    directions = trial.directions(rows,w,[0,1])
    h,_,_ = trial.group.forward_hidden(base,x,m,correction,writes=directions,radius_fraction=ramp)
    h[1:, :,0].sum().backward()
    assert torch.isfinite(w.grad).all() and w.grad.norm()>0.
    assert all(p.grad is None for p in base.parameters())
