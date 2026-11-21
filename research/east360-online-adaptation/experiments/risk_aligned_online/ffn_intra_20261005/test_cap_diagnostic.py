import numpy as np
import pytest

from ..representation_analysis import threshold_diagnostic
from .cap_diagnostic import peak_caps


def test_compressed_cap_scan_matches_full_curve_with_ties_and_tardy():
    rows=[]
    for i,(d,scores) in enumerate([(True,[.3,.4,.9]),(True,[.2,.4,.3]),
                                  (True,[.1,.2,.99]),(False,[.2,.3,.4]),
                                  (False,[.1,.3,.2]),(False,[.2,.2,.2])]):
        rows.append(dict(shot=i,is_disrupt=d,down=1.,times=np.array([.90,.97,1.]),scores=np.array(scores)))
    caps=(0.,1/3,.5,1.)
    slow=threshold_diagnostic(rows,caps=caps)
    fast=peak_caps(rows,caps)
    for key,expected in slow['caps'].items():
        for field in ('tp','fp','n_disrupt','n_safe','threshold','tpr','fpr','J'):
            assert fast['caps'][key][field]==pytest.approx(expected[field])


def test_no_timely_positive_is_kept_in_denominator():
    rows=[dict(shot=1,is_disrupt=True,down=1.,times=np.array([1.]),scores=np.array([1.])),
          dict(shot=2,is_disrupt=False,down=None,times=np.array([.1]),scores=np.array([.2]))]
    best=peak_caps(rows,(.2,))['caps']['0.20']
    assert best['n_disrupt']==1 and best['tp']==0 and best['fp']==0
