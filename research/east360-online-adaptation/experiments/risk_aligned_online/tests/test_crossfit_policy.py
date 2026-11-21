import numpy as np
import pytest
from unittest.mock import patch

from experiments.risk_aligned_online.crossfit_teacher_20261025 import learning


def test_four_equal_folds_match_twenty_shot_objective_and_gradient():
    from experiments.risk_aligned_online.operational_bag_loss_20261025.learning import aggregate
    rows=[];responses=[];teachers=[]
    for fold in range(4):
        local=[dict(shot=5*fold+j,flag=j==0,delta=(j-2)*.01*(fold+1),
                    gradient=np.array([j+1.,fold+1.])) for j in range(5)]
        rows.extend(local)
        loss,gradient,details=aggregate(local)
        counts=dict(tp=fold%2,fp=1,very_early_over3000=0)
        responses.append((loss,gradient,details,dict(current=counts,baseline=counts)))
        teachers.append(dict(model=None,shots={r['shot']:r for r in local}))
    expected,grad,details=aggregate(rows)
    with patch.object(learning,'evaluate_policy',side_effect=responses):
        value,g,got,gate=learning.evaluate_teachers(teachers,None,.1254,device='cpu')
    assert value==pytest.approx(expected)
    np.testing.assert_allclose(g,grad)
    assert {r['shot'] for r in got}=={r['shot'] for r in details}
    assert gate['current']==dict(tp=2,fp=4,very_early_over3000=0)
    assert not gate['passed']  # mean safety harm cannot be concealed by positive improvement
    with pytest.raises(ValueError):learning.evaluate_teachers(teachers[:3],None,.1254,device='cpu')
    teachers[1]['shots']=teachers[0]['shots']
    with pytest.raises(ValueError):learning.evaluate_teachers(teachers,None,.1254,device='cpu')


def test_runtime_names_preserve_underlying_arrays_and_require_all_arms():
    payload={a:np.arange(3)+i for i,a in enumerate(learning.RENAME)}
    got=learning.rename_arms(payload)
    assert set(got)=={'off',*learning.ARMS}
    for old,new in learning.RENAME.items():assert got[new] is payload[old]
    with pytest.raises(ValueError):learning.rename_arms({'unexpected':np.arange(3)})


def test_gate_requires_crossfit_not_insample_win_and_history_protection():
    base=dict(tp=1,fp=1,very_early_over3000=0,bag_disrupt_loss=.5,bag_safe_loss=.3,bag_balanced_loss=.4)
    results={a:dict(base) for a in ('off',)+learning.ARMS}
    results['crossfit_context'].update(bag_disrupt_loss=.4,bag_safe_loss=.2,bag_balanced_loss=.3)
    assert learning.eligible(results,True)
    assert not learning.eligible(results,False)
    results['uniform']['bag_balanced_loss']=.29
    assert not learning.eligible(results,True)
