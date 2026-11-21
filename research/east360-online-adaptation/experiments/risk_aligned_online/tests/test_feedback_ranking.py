import numpy as np
import pytest
from experiments.risk_aligned_online.feedback_capacity_20261024.ranking import weighted_auc


def test_weighted_auc_matches_pair_enumeration_with_ties():
    y=np.array([0,1,0,1,1]);s=np.array([.1,.3,.3,.2,.3]);w=np.array([2.,1.,3.,2.,4.])
    pairs=[(w[i]*w[j],float(s[i]>s[j])+.5*float(s[i]==s[j])) for i in range(5) for j in range(5) if y[i]==1 and y[j]==0]
    expected=sum(a*b for a,b in pairs)/sum(a for a,b in pairs)
    assert abs(weighted_auc(y,s,w)-expected)<1e-12
    assert weighted_auc(y,s*3+7,w)==pytest.approx(expected)


def test_duplication_with_split_weights_preserves_auc():
    y=np.array([0,1,0,1]);s=np.array([1.,2.,3.,4.]);w=np.ones(4)
    assert weighted_auc(np.repeat(y,2),np.repeat(s,2),np.repeat(w/2,2))==weighted_auc(y,s,w)


def test_auc_rejects_missing_class():
    with pytest.raises(ValueError):weighted_auc(np.ones(3),np.arange(3),np.ones(3))
