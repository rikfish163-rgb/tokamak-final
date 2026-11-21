import copy
import pytest

from ..east_joint_feedback_20261023.audit import check_training, deterministic_history
from ..east_joint_feedback_20261023.run import gate


def test_preparation_receipts_reject_wrong_loss_and_foreign_identity():
    entries=[dict(step=0,episodes=[dict(shot=1,disruptive=False,count=256,ends=[260]),
                                    dict(shot=2,disruptive=True,count=384,ends=[400])])]
    row=dict(step=0)
    for arm in ('joint','meta'):
        objective=.4+.2*.1 if arm=='joint' else .5*(.4+.2)+.2*.1
        row[arm]=dict(seconds=3.,grad_norm=.5,group_grad_norms=dict(encoder=.4,head=.3),
            episodes=[dict(**e,risk_before=.4,risk_after=.2,forecast=.1,objective=objective,
                           all_counts=list(range(256,e['count']+1,128))) for e in entries[0]['episodes']])
    assert check_training(entries,[row],{1,2})==dict(episodes=4,forward_steps=6)
    with pytest.raises(AssertionError):check_training(entries,[row],{1})
    bad=copy.deepcopy(row);bad['meta']['episodes'][0]['objective']+=.1
    with pytest.raises(AssertionError):check_training(entries,[bad],{1,2})
    other=copy.deepcopy(row);other['meta']['seconds']=8.
    assert deterministic_history([row])==deterministic_history([other])
    assert 'seconds' in row['meta']


def test_intra_gate_rejects_offline_only_gain_and_weakened_start():
    def metric(tp,fp):return dict(tp=tp,fp=fp,fpr=fp/16,J=tp/4-fp/16)
    r=dict(warm_static=metric(3,1),joint_static=metric(3,1),joint_intra=metric(3,1),
           meta_static=metric(3,1),meta_intra=metric(4,1))
    assert gate(r)
    r['meta_static']=metric(4,1)
    assert not gate(r)
    r['meta_static']=metric(1,1);r['meta_intra']=metric(3,1)
    assert not gate(r)
