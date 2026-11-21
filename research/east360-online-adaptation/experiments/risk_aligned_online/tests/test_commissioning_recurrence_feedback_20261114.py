import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_recurrence_feedback_20261114 as replay


def test_issue_is_prefix_only_and_donated_target_excludes_shift_and_level():
    r = np.random.default_rng(52)
    x = r.normal(size=(505,len(replay.CHANNELS)))
    m = np.ones_like(x,bool)
    ctx = r.normal(size=(len(x),2))
    cm = np.ones_like(ctx,bool)
    q = np.arange(127,len(x),32)
    a = replay.window_view(x,m,ctx,cm,q)
    b = replay.window_view(x[:291],m[:291],ctx[:291],cm[:291],q[q<291])
    n = len(b.key)
    for k in ('key','issue_ok','location','spread'):
        np.testing.assert_array_equal(getattr(a,k)[:n],getattr(b,k))
    assert a.complete[n-1] and not b.complete[-1]
    np.testing.assert_allclose(a.target[a.complete].mean(axis=1),0.,atol=3e-7)
    assert a.issue_ok[-1] and not a.complete[-1]


def test_donated_shape_is_removed_without_replacing_unmatched_values():
    shape = np.zeros((4,5,len(replay.CHANNELS)))
    change = np.linspace(-1,1,5)
    for i in range(len(shape)):
        shape[i] = change[:,None]*(i+1)
    select = np.broadcast_to([[-1,-1,0,1]], (4,4))
    out = replay.transplant(shape, select)
    np.testing.assert_array_equal(out[:2],0.)
    np.testing.assert_allclose(out[2],shape[0])
    np.testing.assert_allclose(out[3],shape[1])
    pred = replay.aligned_difference(shape[2:], out[2:])
    np.testing.assert_allclose(pred[0],shape[2]-shape[0])
    np.testing.assert_allclose(pred[1],shape[3]-shape[1])


def test_earlier_comparison_uses_one_distinct_finished_slot():
    q = np.array([127,159,223,255])
    m = np.array([True, False, True, True])
    ix, keep = replay.previous_issue(q,m)
    np.testing.assert_array_equal(ix, [0,0,2,2])
    np.testing.assert_array_equal(keep,[False,True,False,True])
    # In the second row the prior target is indeed matured but the caller
    # separately requires the current query itself to have a precedent.
    assert (q[ix[keep]]+26 <= q[keep]).all()


def test_no_motion_feedback_and_exact_decomposition_on_the_same_observation():
    target = torch.tensor([[[1.,2.],[-1.,-2.]]])
    forecast = target*.4
    history = torch.stack((target*.1,target*.2,target*.3))
    earlier = target*.5
    steps = dict(zip(replay.NEW,replay.residuals(target,forecast,history,earlier),strict=True))
    torch.testing.assert_close(steps['zero_motion_innovation'],target)
    torch.testing.assert_close(steps['matched_innovation']+steps['matched_explained'],
                               steps['network_innovation'])
    torch.testing.assert_close(steps['earlier_matched_innovation'],earlier)
