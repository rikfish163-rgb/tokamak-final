"""Cohort and clock checks for the small copied-encoder pilot."""

import numpy as np
import pytest
import torch

import run_cta_copy_pilot_20261107 as run


def test_train_selection_is_fixed_stratified_and_without_replacement():
    rows = [dict(shot=i+1000*g, group=group) for g,group in enumerate(run.GROUPS) for i in range(70)]
    a = run.choose_train_rows(rows)
    assert a == run.choose_train_rows(rows)
    assert len(a) == 144 and len({q['shot'] for q in a}) == 144
    assert all(sum(q['group']==group for q in a)==48 for group in run.GROUPS)
    assert all(q in rows for q in a)


def test_group_mean_gives_equal_weight_to_physical_shots():
    rows = [dict(shot=1,group='safe')]*3 + [dict(shot=2,group='safe'),dict(shot=3,group='timely')]
    assert run.group_mean([1,1,1,9,100],rows,'safe') == 5


@pytest.fixture
def data_api(monkeypatch):
    signal = dict(values=np.ones((200,23),np.float32), available=np.ones((200,23),bool),
                  times=np.arange(200)/1000,device='J-TEXT',shot=1)
    monkeypatch.setattr(run.r.simple_data,'source_roles',lambda:dict(train=[1],val=[2],test=[3]))
    monkeypatch.setattr(run.r.simple_data,'read_shot',lambda device,sid:dict(signal,shot=sid,
                        values=signal['values'].copy(),available=signal['available'].copy()))
    monkeypatch.setattr(run.r.io23,'read_outcome',lambda device,sid:(False,None))
    row=dict(shot=1,origin=127,release=153,query=169,query_time=.169,
             disruptive=False,group='safe',label=0,source_horizon_ms=25.)
    return row


def test_loader_uses_fixed_causal_slices_and_target_channel_mask(data_api):
    loaded=run.load_windows([data_api],'train','cpu')
    assert loaded['x0'].shape==(1,128,23)
    assert loaded['truth'].shape==(1,25,23)
    assert not loaded['mask'][:,:,[2,10]].any()
    assert not loaded['a0'][:,:,2].any()
    assert loaded['mask'][:,:,0].all()


def test_loader_rejects_other_cohort_before_reading(data_api):
    with pytest.raises(ValueError,match='outside declared cohort'):
        run.load_windows([dict(data_api,shot=2)],'train','cpu')


def test_loader_rejects_early_release(data_api):
    with pytest.raises(ValueError,match='maturity clock'):
        run.load_windows([dict(data_api,release=150)],'train','cpu')


def test_loader_rejects_changed_label(data_api):
    with pytest.raises(ValueError,match='label contract'):
        run.load_windows([dict(data_api,label=1)],'train','cpu')


def test_gradient_summary_does_not_turn_unmeasured_teacher_steps_into_zero():
    from run_cta_copy_gradient_20261107 import summarize
    rows=[dict(shot=i,group=g,cosine=.1,zero_gradient=False) for i,g in enumerate(run.GROUPS)]
    assert all(v['actual_full_step_harmful_windows'] is None for v in summarize(rows).values())
