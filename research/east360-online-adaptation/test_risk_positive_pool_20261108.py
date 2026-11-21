from copy import deepcopy

import numpy as np
import pytest
import torch

from test_joint_stream_20261107 import signal


def source_shots():
    shots=[]
    for sid in (1,2,3):
        s=signal(700)
        s.update(shot=sid,valid=np.ones(700,bool),y=(np.arange(700)>=400).astype(np.float32))
        shots.append(s)
    return shots


def note(shot,first,group):
    origins=[first+32*k for k in range(4)];query=origins[-1]+26
    return dict(shot=shot,origins=origins,query=query,label_time=query+25,group=group)


def test_positive_union_is_deduplicated_per_shot_and_does_not_include_negatives():
    from run_risk_positive_pool_20261108 import positive_origins

    pools={'0_to_0':[(9,np.array([127]))],
           '0_to_1':[(0,np.array([127,128])),(1,np.array([150]))],
           '1_to_1':[(0,np.array([128,159])),(2,np.array([200]))]}
    union=positive_origins(pools)
    assert [i for i,_ in union]==[0,1,2]
    np.testing.assert_array_equal(union[0][1],[127,128,159])
    assert 9 not in [i for i,_ in union]


def test_resampling_preserves_every_negative_and_future_label_at_each_step():
    from run_risk_teacher_20261107 import teacher_pools
    from run_risk_positive_pool_20261108 import replace_positive_notes

    shots=source_shots();pools=teacher_pools(shots,steps=4)
    old=[note(1,127,'0_to_0'),note(1,270,'0_to_1'),note(1,300,'1_to_1'),note(2,140,'0_to_0')]
    saved=deepcopy(old)
    new=replace_positive_notes(shots,pools,old,seed=141)
    assert old==saved
    assert new[0]==old[0] and new[3]==old[3]
    assert new==replace_positive_notes(shots,pools,old,seed=141)
    assert [r['group'].endswith('_to_1') for r in new]==[False,True,True,False]
    for r in new:
        s=shots[r['shot']-1]
        assert float(s['y'][r['label_time']])==int(r['group'].split('_to_')[1])
        assert r['query']==r['origins'][-1]+26


def test_fixed_sequence_builder_keeps_values_masks_and_rejects_a_changed_native_label():
    from risk_meta_sequence_20261108 import sequence_from_note

    s=source_shots()[0];r=note(1,270,'0_to_1')
    es,qx,qa,label,_=sequence_from_note(s,r,'cpu')
    assert len(es)==4 and label.tolist()==[1.]
    torch.testing.assert_close(qx,torch.from_numpy(s['values'][r['query']-127:r['query']+1])[None])
    assert not qa[...,2].any()
    for o,e in zip(r['origins'],es,strict=True):
        torch.testing.assert_close(e['truth'],torch.from_numpy(s['values'][o+1:o+26])[None])
        assert not e['mask'][...,2].any() and not {'y','down','label'}&set(e)
    with pytest.raises(ValueError,match='native'):
        sequence_from_note(s,dict(r,group='1_to_1'),'cpu')


def test_sampling_arms_share_forward_state_and_do_not_execute_extra_feedbacks():
    from run_risk_teacher_20261107 import comparison_curves
    from test_risk_feature_intra_20261107 import setup

    base,decoder,_=setup()
    models={k:deepcopy(decoder) for k in ('recorded_sampling','pooled_sampling')}
    curves,info=comparison_curves(base,models,signal(164))
    assert set(curves)=={'static','recorded_sampling_real','pooled_sampling_real'}
    assert all(v['updates']=={'real':1} for v in info.values())
    np.testing.assert_array_equal(curves['recorded_sampling_real'],curves['pooled_sampling_real'])
