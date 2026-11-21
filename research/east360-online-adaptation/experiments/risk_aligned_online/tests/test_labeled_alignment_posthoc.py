import numpy as np

from experiments.risk_aligned_online.labeled_alignment_20261023.posthoc import updated_region_masks


def test_safe_change_region_does_not_modify_full_shot_evaluation_mask():
    valid=np.ones(400,dtype=bool)
    masks=updated_region_masks(False,None,valid)
    np.testing.assert_array_equal(valid,np.ones(400,dtype=bool))
    assert not masks['safe'][:256].any()
    assert masks['safe'][256:].all()
    assert not masks['disrupt_early'].any()


def test_timely_change_region_is_strictly_over30_and_at_most125():
    lead=np.full(400,200.,dtype=float)
    lead[256:261]=[126.,125.,31.,30.,-1.]
    valid=lead>=0
    before=valid.copy()
    masks=updated_region_masks(True,lead,valid)
    np.testing.assert_array_equal(valid,before)
    np.testing.assert_array_equal(masks['disrupt_timely_window'][256:261],[False,True,True,False,False])
    assert not masks['disrupt_early'][:256].any()
