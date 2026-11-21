from copy import deepcopy

import pytest

from experiments.risk_aligned_online.verify_operational_memory import flips,verify_receipt,verify


def test_flips_separates_gross_rescues_from_gross_damage():
    old=[dict(shot=1,is_disrupt=True,timely=False,alarm=False),
         dict(shot=2,is_disrupt=True,timely=True,alarm=True),
         dict(shot=3,is_disrupt=False,timely=False,alarm=False),
         dict(shot=4,is_disrupt=False,timely=False,alarm=True)]
    new=deepcopy(old);new[0]['timely']=True;new[1]['timely']=False
    new[2]['alarm']=True;new[3]['alarm']=False
    assert flips(old,new)==dict(rescued_disruptive=[1],lost_disruptive=[2],
        removed_false_alarm=[4],new_false_alarm=[3])
    with pytest.raises(AssertionError):flips(old,new[::-1])


def test_first30_cannot_apply_a_persistent_proposal():
    verify_receipt(dict(accepted=False,actual_steps=0),index=20,ended_ids=list(range(30)))
    with pytest.raises(AssertionError):
        verify_receipt(dict(accepted=True,actual_steps=1),index=20,ended_ids=list(range(30)))


def test_skipped_update_still_requires_exact_causal_history():
    row=dict(effective_from_index=32,full_guard_replay=True,history_ids=list(range(31)),
             actual_steps=0,accepted=False)
    verify_receipt(row,index=31,ended_ids=list(range(40)))
    row['history_ids'].append(31)
    with pytest.raises(AssertionError):verify_receipt(row,index=31,ended_ids=list(range(40)))


def test_incomplete_experiments_are_not_silently_certified(tmp_path):
    with pytest.raises(ValueError,match='not complete'):verify(tmp_path)
    with pytest.raises(ValueError,match='completed control'):verify(tmp_path,completed_only=True)
