import numpy as np
import pytest
import torch

from test_joint_stream_20261107 import signal


def test_teacher_source_pools_require_all_future_times_and_keep_native_groups():
    from run_risk_teacher_20261107 import teacher_pools

    s=signal(240)
    s['valid']=np.arange(240)<230
    s['y']=(np.arange(240)>=190).astype(np.float32)
    pools=teacher_pools([s])
    for name,rows in pools.items():
        assert rows
        for i,origins in rows:
            assert i==0
            for origin in origins:
                assert s['valid'][origin-127:origin+52].all()
                assert name==f"{int(s['y'][origin+26])}_to_{int(s['y'][origin+51])}"
    assert pools['1_to_1'][0][1][-1]==178


def test_sampled_teacher_query_contains_only_arrived_signal_not_the_outer_label_time():
    from run_risk_teacher_20261107 import teacher_pools,teacher_episode

    s=signal(240)
    s['valid']=np.ones(240,bool)
    s['y']=(np.arange(240)>=190).astype(np.float32)
    e,qx,qa,labels,note=teacher_episode([s],teacher_pools([s]),np.random.default_rng(7),'0_to_1','cpu')
    o=note['origin'];q=note['query']
    assert q==o+26 and note['label_time']==o+51
    torch.testing.assert_close(e['truth'],torch.from_numpy(s['values'][o+1:o+26])[None])
    torch.testing.assert_close(qx,torch.from_numpy(s['values'][q-127:q+1])[None])
    assert not e['a'][...,2].any() and not e['mask'][...,2].any() and not qa[...,2].any()
    assert labels.tolist()==[1.]
    assert not {'y','label','down','is_disrupt'} & set(e)


def test_single_control_feedback_does_not_execute_other_arms():
    from run_risk_feature_intra_20261107 import feature_curves
    from test_risk_feature_intra_20261107 import setup

    base,decoder,_=setup()
    curves,info=feature_curves(base,decoder,signal(164),feedbacks=('real',))
    assert set(curves)=={'static','real'}
    assert info['updates']=={'real':1}


def test_continuous_credit_comparison_runs_only_the_requested_same_stream_arms():
    from copy import deepcopy
    from run_risk_teacher_20261107 import comparison_curves
    from test_risk_feature_intra_20261107 import setup

    base,decoder,_=setup()
    models={name:deepcopy(decoder) for name in ('control','last_step','taught')}
    curves,info=comparison_curves(base,models,signal(164))
    assert set(curves)=={'static','control_real','last_step_real','taught_real','taught_wrong_time','taught_hold'}
    for name in ('control','last_step'):
        assert info[name]['updates']=={'real':1}
        np.testing.assert_allclose(curves[f'{name}_real'],curves['taught_real'],atol=3e-6,rtol=0)
    assert info['taught']['updates']=={'real':1,'wrong_time':1,'hold':1}
    cold,_=comparison_curves(base,models,signal(164),warm=True)
    for value in cold.values():
        np.testing.assert_array_equal(value,cold['static'])


def test_primary_continuous_comparison_does_not_run_unrequested_feedback_controls():
    from copy import deepcopy
    from run_risk_teacher_20261107 import comparison_curves
    from test_risk_feature_intra_20261107 import setup

    base,decoder,_=setup()
    models={name:deepcopy(decoder) for name in ('control','last_step','taught')}
    curves,info=comparison_curves(base,models,signal(164),feedback_controls=False)
    assert set(curves)=={'static','control_real','last_step_real','taught_real'}
    assert all(v['updates']=={'real':1} for v in info.values())


def test_followup_feedback_controls_only_execute_the_selected_decoder_and_feedbacks():
    from run_risk_teacher_20261107 import comparison_curves
    from test_risk_feature_intra_20261107 import setup

    base,decoder,_=setup()
    plan={'last_step':('wrong_time','hold')}
    curves,info=comparison_curves(base,{'last_step':decoder},signal(164),feedback_plan=plan)
    assert set(curves)=={'static','last_step_wrong_time','last_step_hold'}
    assert info['last_step']['updates']=={'wrong_time':1,'hold':1}


def control_archives(tmp_path):
    import run_joint_trajectory_20261107 as j

    parent=tmp_path/'primary';follow=tmp_path/'follow'
    for path in (parent,follow):
        path.mkdir();(path/'shots').mkdir()
        j.r.write_new(path/'protocol.json',dict(order=[1,2],fit='same-fitted-decoder',
            threshold=.7,warm=20,parent_sha256='same-locked-parent'))
    times=np.array([.8,.9,1.])
    truths=[dict(shot=1,disruptive=True,down=1.),dict(shot=2,disruptive=False,down=None)]
    static=[np.array([.2,.6,.4]),np.array([.2,.2,.2])]
    real=[np.array([.2,.9,.9]),np.array([.2,.2,.2])]
    wrong=[static[0],np.array([.2,.8,.4])]
    rows_parent={'static':[],'last_step_real':[]}
    rows_follow={'static':[],'last_step_wrong_time':[],'last_step_hold':[]}
    for i,truth in enumerate(truths):
        for name,p in [('static',static[i]),('last_step_real',real[i])]:
            rows_parent[name].append(j.score(times,p,truth,.7))
        for name,p in [('static',static[i]),('last_step_wrong_time',wrong[i]),('last_step_hold',real[i])]:
            rows_follow[name].append(j.score(times,p,truth,.7))
        np.savez(parent/'shots'/f'{i+1:04d}_{i+1}.npz',static=static[i],last_step_real=real[i])
        np.savez(follow/'shots'/f'{i+1:04d}_{i+1}.npz',static=static[i],last_step_wrong_time=wrong[i],last_step_hold=real[i])
    for path,rows in [(parent,rows_parent),(follow,rows_follow)]:
        j.r.write_new(path/'result.json',dict(status='HDF5_RISK_TEACHER360',rows=rows,
            metrics={name:j.counts(v) for name,v in rows.items()}))
    return parent,follow


def test_followup_comparison_uses_the_saved_real_predictions_not_a_new_unexecuted_arm(tmp_path):
    import run_joint_trajectory_20261107 as j
    from run_risk_teacher_20261107 import compare_controls

    parent,follow=control_archives(tmp_path)
    compare_controls(follow,parent)
    result=j.r.read(follow/'paired_real_controls.json')
    assert result['real_vs_feedback']['wrong_time']['tp_gained']==[1]
    assert result['real_vs_feedback']['wrong_time']['fp_removed']==[2]
    assert result['real_vs_feedback']['hold']==dict(tp_gained=[],tp_lost=[],fp_added=[],fp_removed=[])
    assert result['maximum_static_error']==0


def test_followup_comparison_rejects_a_different_fitted_parent(tmp_path):
    import run_joint_trajectory_20261107 as j
    from run_risk_teacher_20261107 import compare_controls

    parent,follow=control_archives(tmp_path)
    p=j.r.read(follow/'protocol.json');p['fit']='different-fitted-decoder'
    (follow/'protocol.json').write_text(__import__('json').dumps(p))
    with pytest.raises(ValueError,match='same'):
        compare_controls(follow,parent)
