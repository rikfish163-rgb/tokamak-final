import pytest
from experiments.risk_aligned_online.preparation_factorial_20261023.audit import contrasts


def test_paired_differences_do_not_call_a_sensitivity_loss_an_intra_gain():
    stats={f'{c}_{a}':dict(tp=2,fp=3,n_disrupt=4,n_safe=16,tpr=.5,fpr=.1875,J=.3125)
           for c in ('raw_bce','raw_moments','mapped_bce','mapped_moments') for a in ('static','intra')}
    stats['raw_bce_intra'].update(tp=1,fp=1,tpr=.25,fpr=.0625,J=.1875)
    stats['mapped_bce_intra'].update(fp=2,fpr=.125,J=.375)
    r=contrasts(stats)
    assert r['intra']['raw_bce']['delta_tp']==-1 and not r['intra']['raw_bce']['screen_pass']
    assert r['intra']['mapped_bce']['delta_fp']==-1 and r['intra']['mapped_bce']['screen_pass']
    assert r['density_map']['bce_intra']['delta_tp']==1
    assert r['interaction']['static']['delta_J']==0
    with pytest.raises(ValueError):contrasts({})
