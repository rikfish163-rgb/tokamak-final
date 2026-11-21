import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.horizon_memory_20261023.learning import (
    FoldAccess, make_memory, predict, gate,
)
from experiments.risk_aligned_online.learned_update_loss_20261023.learning import folds
from experiments.risk_aligned_online.tests.test_task_geometry import small_model


def records():
    rng = np.random.default_rng(131)
    result = []
    for sid in range(20):
        d = sid < 4
        result.append(dict(shot=sid, device='EAST', is_disrupt=d, down=.6 if d else None,
            values=rng.normal(size=(600,23)).astype('float32'), available=np.ones((600,23),bool),
            times=np.arange(600)/1000, valid=np.ones(600,bool),
            y=np.r_[np.zeros(475),np.ones(125)] if d else np.zeros(600)))
    return result


def test_folds_fit_all_preprocessing_separately_and_gate_forbids_held_labels():
    rows=records();plans=folds(rows)
    assert sorted(i for p in plans for i in p['held'])==list(range(20))
    for p in plans:
        assert len(p['fit'])==15 and len(p['held'])==5 and not set(p['fit'])&set(p['held'])
        access=FoldAccess(p,[100,101]);access.check('signal','J-TEXT',100)
        access.check('outcome','EAST',p['fit'][0])
        with pytest.raises(AssertionError):access.check('signal','EAST',p['held'][0])
        with pytest.raises(AssertionError):access.check('outcome','EAST',p['held'][0])
        access.phase='predict';access.check('signal','EAST',p['held'][0])
        with pytest.raises(AssertionError):access.check('outcome','EAST',p['held'][0])
        with pytest.raises(AssertionError):access.check('signal','EAST',99999)


def test_memory_keys_pair_past_context_with_same_later_training_window():
    model=small_model();rows=records();plan=folds(rows)[0]
    memory,manifest=make_memory(model,[r for r in rows if r['shot'] in plan['fit']],device='cpu')
    assert len(manifest)==480 and set(memory['shots'])==set(plan['fit'])
    assert memory['keys'].shape==memory['lag_keys'].shape==(480,16)
    assert all(r['key_end_now']==r['end'] and r['key_end_lag']==r['end']-25 for r in manifest)
    assert all(r['key_end_lag']>=127 for r in manifest)
    assert not np.array_equal(memory['keys'],memory['lag_keys'])
    with pytest.raises(ValueError):make_memory(model,rows[:2]+rows[:1],device='cpu')


def test_stream_is_causal_resets_each_shot_and_has_actual_encoder_updates():
    model=small_model();rows=records();plan=folds(rows)[0]
    memory,_=make_memory(model,[r for r in rows if r['shot'] in plan['fit']],device='cpu')
    signal={k:rows[0][k][:400].copy() for k in ('values','available')}
    origin={k:v.clone() for k,v in model.state_dict().items()}
    scores,updates=predict(model,signal,memory,device='cpu')
    altered={k:v.copy() for k,v in signal.items()};altered['values'][256:]=100
    again,other=predict(model,altered,memory,device='cpu')
    repeat,repeated_updates=predict(model,signal,memory,device='cpu')
    for name in scores:
        np.testing.assert_array_equal(scores[name][:256],again[name][:256])
        np.testing.assert_array_equal(scores[name],repeat[name])
        np.testing.assert_array_equal(scores[name][:256],scores['static'][:256])
    assert updates==repeated_updates
    for arm in updates:
        assert updates[arm][0]==other[arm][0]
        assert updates[arm][0]['last_observation']==255
        assert updates[arm][0]['first_affected_prediction']==256
        assert updates[arm][0]['parameter_delta_norm']>0
    for j in range(len(updates['now_rank'])):
        lengths=[updates[a][j]['parameter_delta_norm'] for a in ('now_rank','lag_rank','initial_rank')]
        assert max(lengths)-min(lengths)<2e-7
    for k,v in model.state_dict().items():torch.testing.assert_close(v,origin[k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(model,dict(signal,y=np.zeros(400)),memory,device='cpu')


def test_promotion_requires_real_warning_gain_and_live_context_contribution():
    results={a:dict(tp=3,fp=2,n_disrupt=4,n_safe=16,J=.625,fpr=.125)
             for a in ('static','forecast','now_rank','lag_rank','initial_rank')}
    assert not gate(results)
    results['lag_rank'].update(tp=4,J=.875)
    assert gate(results)
    results['initial_rank'].update(tp=4,J=.875)
    assert not gate(results)
    results['initial_rank'].update(tp=3,J=.625)
    results['lag_rank'].update(fp=3,fpr=.1875,J=.8125)
    assert not gate(results)  # New false alarms violate this preregistered screen.
