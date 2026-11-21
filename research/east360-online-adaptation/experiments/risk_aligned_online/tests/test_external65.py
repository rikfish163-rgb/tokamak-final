import numpy as np
import pytest

from ..external65_20261023.analysis import partition, paired_changes, interval_peaks


def test_partition_excludes_all_preparation_identities_and_rejects_test_overlap():
    rows=[dict(shot=i,is_disrupt=i%2==0) for i in range(1,8)]
    folds=[dict(fit=[1,2],held=[3]),dict(fit=[1,3],held=[2])]
    assert [r['shot'] for r in partition(rows,folds,[99])]==[4,5,6,7]
    with pytest.raises(ValueError):partition(rows,folds,[5])
    with pytest.raises(ValueError):partition(rows+rows[:1],folds,[99])
    with pytest.raises(ValueError):partition(rows,[dict(fit=[1],held=[1])],[99])


def test_pairing_uses_shot_identity_and_reports_both_sides():
    before=[dict(shot=1,is_disrupt=True,timely=True,alarm=True),
            dict(shot=2,is_disrupt=False,timely=False,alarm=False),
            dict(shot=3,is_disrupt=True,timely=False,alarm=False)]
    after=[dict(before[2],timely=True,alarm=True),dict(before[0],timely=False),
           dict(before[1],alarm=True)]
    assert paired_changes(before,after)==dict(rescued=[3],lost=[1],removed_fp=[],added_fp=[2])
    with pytest.raises(ValueError):paired_changes(before,after[:-1])
    with pytest.raises(ValueError):paired_changes(before,[dict(r,is_disrupt=False) for r in after])


def test_timely_peak_excludes_exact30_and_postterminal_but_includes_early():
    t=np.array([0.,1.,1.875,1.969,1.970,2.,2.001])
    p=np.array([.1,.8,.4,.5,.9,.99,1.])
    result=interval_peaks(t,p,True,2.)
    assert result==dict(max_valid=.99,max_timely=.8,max_label_timely=.5)
    assert interval_peaks(t,p,False,None)==dict(max_valid=1.,max_timely=None,max_label_timely=None)
    with pytest.raises(ValueError):interval_peaks(t,p[:-1],True,2.)
