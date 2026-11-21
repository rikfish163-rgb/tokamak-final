import numpy as np
import pytest


def rows():
    rng=np.random.default_rng(42)
    return [dict(shot=s,is_disrupt=s%2==0,context=rng.normal(size=185).tolist(),
                 risk_alignment=rng.normal(size=23).tolist()) for s in range(4) for _ in range(3)]


def test_policy_exact_allowed_id_check_and_equal_class_shot_weight():
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.policy import fit, predicted_alignment
    data=rows(); p=fit(data,[0,1,2,3])
    assert p['allowed_ids']==[0,1,2,3]
    assert np.isclose(sum(p['weights']),1.)
    for s in range(4):
        assert sum(w for w,r in zip(p['weights'],data) if r['shot']==s)==pytest.approx(.25)
    np.testing.assert_allclose(predicted_alignment(p,p['mean']),p['target_mean'],atol=1e-10)
    with pytest.raises(ValueError): fit(data,[0,1,2])
    with pytest.raises(ValueError): fit(data+[dict(data[0],is_disrupt=False)],[0,1,2,3])


def test_shuffled_teacher_repeatable_not_context_shuffle():
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.policy import fit
    a=fit(rows(),[0,1,2,3],shuffle=True);b=fit(rows(),[0,1,2,3],shuffle=True)
    assert a==b and a['shuffle_order']!=list(range(12))
    normal=fit(rows(),[0,1,2,3])
    assert a['mean']==normal['mean'] and a['scale']==normal['scale']
    assert a['coefficients']!=normal['coefficients']


def test_policy_abstains_on_no_positive_predicted_benefit():
    from experiments.risk_aligned_online.feedback_cone_feasibility_20261024.policy import decide
    w,receipt=decide(np.array([-2.,-1.]),np.ones(2))
    assert np.array_equal(w,np.zeros(2)) and receipt['abstain']
    w,receipt=decide(np.array([2.,-1.]),np.array([-1.,2.]))
    np.testing.assert_allclose(w,[2/3,1/3],atol=1e-8)
    assert not receipt['abstain']
