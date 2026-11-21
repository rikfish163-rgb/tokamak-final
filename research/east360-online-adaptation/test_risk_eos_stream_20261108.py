"""Ended-only task construction and shared-parent adaptive decoder inference."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from test_joint_stream_20261107 import signal
from test_risk_meta_step_20261107 import teacher_case


def shot(sid,flag):
    s=dict(signal(600),shot=sid,device='EAST',times=np.arange(600)*.001)
    return s,(flag,.550 if flag else None)


def test_incremental_pool_does_not_fabricate_missing_risk_classes():
    from run_risk_teacher_20261107 import teacher_pools

    s=dict(signal(600),valid=np.ones(600,bool),y=np.zeros(600,np.float32))
    with pytest.raises(ValueError,match='all three'):
        teacher_pools([s],steps=4)
    pools=teacher_pools([s],steps=4,require_all_groups=False)
    assert pools['0_to_0'] and not pools['0_to_1'] and not pools['1_to_1']


def test_ended_EAST_tasks_keep_native125_labels_maturity_and_timely_queries():
    from run_risk_eos_stream_20261108 import EndedTasks

    bank=EndedTasks(seed=143)
    for sid,flag in ((1,False),(2,True)):
        s,outcome=shot(sid,flag);bank.add(s,outcome)
    assert [s['shot'] for s in bank.views]==[1,2]
    assert [s['y'].sum() for s in bank.views]==[0.,126.]
    for group in ('0_to_0','0_to_1','1_to_1'):
        es,qx,qa,label,note=bank.draw(group,'cpu')
        assert len(es)==4 and note['label_time']==note['query']+25
        s=bank.views[note['shot']-1];q=note['query']
        assert label.tolist()==[float(s['y'][q+25])]
        assert q==note['origins'][-1]+26
        if note['shot']==2:
            assert round((.550-s['times'][q])*1000,9)>30
        for origin,e in zip(note['origins'],es,strict=True):
            assert origin+25<q and not {'y','down','label'}&set(e)
            torch.testing.assert_close(e['truth'],torch.from_numpy(s['values'][origin+1:origin+26])[None])
        assert not qa[...,2].any()
    foreign,_=shot(3,True);foreign['device']='J-TEXT'
    with pytest.raises(ValueError,match='EAST'):
        bank.add(foreign,(True,.550))


def test_parallel_decoders_do_not_modify_their_shared_persistent_parent():
    from run_risk_eos_stream_20261108 import parallel_curves

    model,_=teacher_case();base=model.risk.eval().requires_grad_(False)
    models={name:deepcopy(model.decoder).eval().requires_grad_(False) for name in ('fixed','signal','guided')}
    before=deepcopy(base.state_dict())
    traces,info=parallel_curves(base,models,signal(190))
    assert set(traces)=={'inter','fixed','signal','guided'}
    for name in ('signal','guided'):
        np.testing.assert_array_equal(traces[name],traces['fixed'])
    assert all(v['updates']=={'real':2} for v in info.values())
    for key,value in base.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_EOS_teacher_runs_only_after_seal_and_preserves_the_recorded_risk_process(tmp_path,monkeypatch):
    import run_joint_trajectory_20261107 as j
    import run_risk_feature_joint_20261108 as joint
    import run_risk_eos_stream_20261108 as eos
    import risk_eos_teacher_20261108 as teacher
    from run_risk_feature_intra_20261107 import feature_curves
    from torch.nn.attention import sdpa_kernel,SDPBackend

    model,_=teacher_case();base=model.risk.eval().requires_grad_(False)
    decoder=model.decoder.eval().requires_grad_(False);order=list(range(1,23))
    inputs={sid:shot(sid,sid%2==1)[0] for sid in order}
    fitted=tmp_path/'fit';fitted.mkdir()
    torch.save(decoder.state_dict(),fitted/'prepared_recorded_sampling.pt')
    source_run=tmp_path/'source';source_run.mkdir();(source_run/'shots').mkdir()
    j.r.write_new(source_run/'protocol.json',dict(order=order,fit=str(fitted),warm=20,
        parent_sha256=j.PARENT_SHA256,threshold=j.r.THRESHOLD,deterministic_algorithms=True,
        source_files={str(j.r.ROOT/name):(j.r.ROOT/name).read_text() for name in
            ('run_risk_feature_intra_20261107.py','risk_feature_intra_20261107.py')}))
    j.r.write_new(source_run/'completed.json',dict(status='SEALED_RISK_TEACHER360',
        records=[dict(shot=sid) for sid in order]))
    torch.use_deterministic_algorithms(True)
    with sdpa_kernel(SDPBackend.MATH):
        for rank,sid in enumerate(order,1):
            traces,_=feature_curves(base,decoder,inputs[sid],warm=rank<=20,feedbacks=('real',))
            j.r.save_arrays(source_run/'shots'/f'{rank:04d}_{sid}.npz',static=traces['static'],
                recorded_sampling_real=traces['real'])
    monkeypatch.setattr(j.r,'configure',lambda:torch.set_num_threads(1))
    monkeypatch.setattr(j.r,'load_parent',lambda _: (deepcopy(base),{'threshold':j.r.THRESHOLD}))
    cfg=dict(order=order,raw_sha256={});archive=dict(sealed=[dict(shot=sid) for sid in order])
    monkeypatch.setattr(j,'cohort',lambda _: (source_run,cfg,archive,('unused','unused',20)))
    monkeypatch.setattr(j,'raw_for',lambda sid,_:inputs[sid])
    monkeypatch.setattr(j.r,'prepare_actual',lambda raw,rank,mapping:dict(raw))
    real_read=j.r.read
    monkeypatch.setattr(j.r,'read',lambda path:dict(calibration_ids=order[:20])
                        if path==j.r.COLD/'input_map.json' else real_read(path))
    active={'out':tmp_path/'reference'};revealed=[]

    def outcome(device,sid):
        assert (active['out']/'shots'/f'{sid:04d}_{sid}.npz').is_file()
        assert revealed==order[:sid-1]
        revealed.append(sid)
        return (sid%2==1,.550 if sid%2 else None)

    def update(slow,shots,records,schedule,initial_weight,**kwargs):
        assert len(revealed)>=21
        assert all(records[i]['shot'] in revealed for draw in schedule for i in draw['indices'])
        with torch.no_grad():
            slow.get_parameter(joint.WEIGHT_KEY).add_(.001)
        return [dict(step=d['step'],loss=0.,risk_loss=0.,prior=0.,gradient_norm=0.) for d in schedule]

    monkeypatch.setattr(j.r.io23,'read_outcome',outcome)
    monkeypatch.setattr(joint,'train_inter',update)
    monkeypatch.setattr(eos,'train_inter',update)
    joint.predict(active['out'],reference_run=source_run,fit_root=fitted,device='cpu')
    parent=active['out'];active['out']=tmp_path/'teacher';revealed.clear()
    real_teach=teacher.teach_decoder;calls=[]

    def teaching(slow,decoder,optimizer,*args,credit):
        assert revealed[-1]>=21
        assert (active['out']/'shots'/f'{revealed[-1]:04d}_{revealed[-1]}.npz').is_file()
        result=real_teach(slow,decoder,optimizer,*args,credit=credit)
        calls.append((revealed[-1],credit))
        return result

    monkeypatch.setattr(teacher,'teach_decoder',teaching)
    eos.predict(active['out'],parent_run=parent,fit_root=fitted,device='cpu')
    assert revealed==order
    assert calls==[(21,'none'),(21,'last'),(22,'none'),(22,'last')]
    completed=real_read(active['out']/'completed.json')
    assert completed['persistent_final_weight_exact'] and completed['eos_updates']==2
    assert completed['source_and_fixed_decoder_unchanged']
    assert completed['maximum_training_error']==0.
    for rank in (21,22):
        item=real_read(active['out']/'eos'/f'{rank:04d}_{rank}.json')
        assert item['task']['shot'] in order[:rank] and item['task_ended_ids']==order[:rank]
