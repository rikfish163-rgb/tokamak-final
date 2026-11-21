import numpy as np
import pytest


def test_phase_score_orients_stages_and_reversing_endpoints_complements_it():
    from experiments.risk_aligned_online.phase_memory_20261024.learning import phase_score
    early=np.array([[0.,0.],[0.,1.],[0.,2.]])
    late=early+np.array([2.,0.]); shots=np.array([1,2,3])
    q=np.array([[0.,1.],[1.,1.],[2.,1.]])
    score,receipt=phase_score(q,early,late,shots)
    assert score[0]<.5 and score[2]>.5
    assert score[0]<score[1]<score[2]
    reverse,_=phase_score(q,late,early,shots)
    np.testing.assert_allclose(reverse,1-score,atol=1e-14)
    np.testing.assert_allclose(receipt['weights'].sum(1),1.)
    assert receipt['temperature']==4.


def test_each_history_shot_has_one_vote_and_constant_coordinates_cannot_identify_phase():
    from experiments.risk_aligned_online.phase_memory_20261024.learning import phase_score
    e=np.zeros((3,2)); l=np.ones((3,2)); q=np.array([[.3,.4]])
    a,_=phase_score(q,e,l,np.array([1,2,3]))
    b,r=phase_score(q,np.repeat(e,[5,1,1],axis=0),np.repeat(l,[5,1,1],axis=0),np.array([1]*5+[2,3]))
    np.testing.assert_allclose(a,b,atol=1e-14)
    assert r['weights'].shape==(1,3)
    with pytest.raises(ValueError,match='phase'):
        phase_score(q,e,e,np.array([1,2,3]))


def test_fold_excludes_both_whole_shots_and_deduplicates_early_queries():
    from experiments.risk_aligned_online.phase_memory_20261024.learning import make_fold
    shots=np.repeat([1,2,3,4,5,6],4); labels=np.repeat([1,1,1,1,0,0],4)
    ends=np.tile([127,127,143,143],6)
    fold=make_fold(shots,labels,ends,1,5)
    assert not np.isin(shots[fold['history']], [1,5]).any()
    assert len(fold['late'])==len(fold['safe'])==4
    assert len(fold['early_unique'])==2
    np.testing.assert_array_equal(ends[fold['early_unique']][fold['pair_to_unique']],ends[fold['late']])


def test_no_promotion_with_one_reversed_disruptive_shot_or_only_mean_level_signal():
    from experiments.risk_aligned_online.phase_memory_20261024.learning import eligible
    def result(v):
        return dict(phase_auc=v,late_safe_auc=.7,paired_order=v,per_disruptive={str(s):dict(phase_auc=v,paired_order=v) for s in range(4)})
    rows={mode:{'phase':result(.7 if mode=='risk' else .6),'late_only':result(.55)}
          for mode in ('risk','mean','reverse','raw_mean','risk_131','risk_132','risk_133')}
    assert eligible(rows)
    rows['risk']['phase']['per_disruptive']['0']['phase_auc']=.4
    assert not eligible(rows)
    rows['risk']['phase']['per_disruptive']['0']['phase_auc']=.7
    rows['mean']['phase']['phase_auc']=.7
    assert not eligible(rows)


def test_auc_matches_independent_roc_with_non_binary_denominators_and_ties():
    from sklearn.metrics import roc_auc_score
    from experiments.risk_aligned_online.phase_memory_20261024.learning import auc
    rng=np.random.default_rng(131)
    for n in (2,4,6,16):
        p=rng.integers(0,12,16);negative=rng.integers(0,12,n)
        expected=roc_auc_score(np.r_[np.ones(16),np.zeros(n)],np.r_[p,negative])
        np.testing.assert_allclose(auc(p,negative),expected,atol=np.finfo('float64').eps,rtol=0)
