"""Shared-parent inference and predict-seal-reveal inter-shot ordering."""

from copy import deepcopy

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from test_risk_feature_intra_20261107 import setup
from test_joint_stream_20261107 import signal


def test_inter_and_both_share_the_exact_parent_without_promoting_fast_weights():
    from run_risk_feature_joint_20261108 import slow_curves
    from run_risk_feature_intra_20261107 import feature_curves

    base,decoder,_=setup();original=deepcopy(base.state_dict())
    expected,_=feature_curves(base,decoder,signal(190),feedbacks=('real',))
    curves,info=slow_curves(base,decoder,signal(190))
    np.testing.assert_array_equal(curves['inter'],expected['static'])
    np.testing.assert_array_equal(curves['both'],expected['real'])
    assert info['updates']=={'real':2}
    for key,value in base.state_dict().items():
        torch.testing.assert_close(value,original[key],atol=0,rtol=0)


def test_stream_reads_each_outcome_only_after_seal_and_updates_only_future_shots(tmp_path,monkeypatch):
    import run_joint_trajectory_20261107 as j
    import run_risk_feature_joint_20261108 as r
    from run_risk_feature_intra_20261107 import feature_curves

    from risk_meta_step_20261107 import MetaOrderedTrajectory

    base,original_decoder,_=setup()
    decoder=MetaOrderedTrajectory(base.center,base.scale,dim=8,scenarios=4)
    decoder.load_state_dict(original_decoder.state_dict())
    decoder.eval().requires_grad_(False)
    initial=deepcopy(base.state_dict());order=list(range(1,23))
    reference=tmp_path/'reference';reference.mkdir();(reference/'shots').mkdir()
    fitted=tmp_path/'fit';fitted.mkdir()
    torch.save(decoder.state_dict(),fitted/'prepared_recorded_sampling.pt')
    signals={sid:dict(signal(340),shot=sid,device='EAST',times=np.arange(340)*.001) for sid in order}
    cfg=dict(order=order,raw_sha256={});archive={'sealed':[dict(shot=sid) for sid in order]}
    protocol=dict(status='RISK_TEACHER_INTRA360',order=order,fit=str(fitted),warm=20,
        parent_sha256=j.PARENT_SHA256,threshold=j.r.THRESHOLD,deterministic_algorithms=True,
        source_files={str(j.r.ROOT/name):(j.r.ROOT/name).read_text() for name in
            ('run_risk_feature_intra_20261107.py','risk_feature_intra_20261107.py')})
    j.r.write_new(reference/'protocol.json',protocol)
    j.r.write_new(reference/'completed.json',dict(status='SEALED_RISK_TEACHER360',
        records=[dict(shot=sid) for sid in order]))
    # The reference cache must use the same deterministic MATH backend as the
    # production cached replay, rather than a different CPU attention path.
    torch.use_deterministic_algorithms(True)
    with sdpa_kernel(SDPBackend.MATH):
        for rank,sid in enumerate(order,1):
            curves,_=feature_curves(base,decoder,signals[sid],warm=rank<=20,feedbacks=('real',))
            j.r.save_arrays(reference/'shots'/f'{rank:04d}_{sid}.npz',static=curves['static'],
                recorded_sampling_real=curves['real'])
    monkeypatch.setattr(j.r,'configure',lambda:torch.set_num_threads(1))
    monkeypatch.setattr(j.r,'load_parent',lambda _: (deepcopy(base),{'threshold':j.r.THRESHOLD}))
    monkeypatch.setattr(j,'cohort',lambda _: (reference,cfg,archive,('unused','unused',20)))
    monkeypatch.setattr(j,'raw_for',lambda sid,_:signals[sid])
    monkeypatch.setattr(j.r,'prepare_actual',lambda raw,rank,mapping:dict(raw))
    real_read=j.r.read
    monkeypatch.setattr(j.r,'read',lambda path:dict(calibration_ids=order[:20])
                        if path==j.r.COLD/'input_map.json' else real_read(path))
    out=tmp_path/'run';revealed=[];trained=[]

    def outcome(device,sid):
        assert device=='EAST'
        assert (out/'shots'/f'{sid:04d}_{sid}.npz').is_file()
        assert revealed==order[:sid-1]
        revealed.append(sid)
        return (sid%2==1,.339 if sid%2 else None)

    def train(model,shots,records,schedule,weight,**kwargs):
        assert len(revealed)>=21 and set(shots)==set(revealed)
        chosen=[records[i] for draw in schedule for i in draw['indices']]
        assert len(chosen)==8*64 and all(v['shot']<=revealed[-1] for v in chosen)
        assert sum(v['shot']==revealed[-1] for v in chosen)==8*32
        for key,value in base.state_dict().items():
            torch.testing.assert_close(value,initial[key],atol=0,rtol=0)
        trained.append(revealed[-1])
        with torch.no_grad():
            model.get_parameter(r.WEIGHT_KEY).add_(.001)
        return [dict(step=d['step'],risk_loss=0.,prior=0.,gradient_norm=0.) for d in schedule]

    monkeypatch.setattr(j.r.io23,'read_outcome',outcome)
    monkeypatch.setattr(r,'train_inter',train)
    r.predict(out,reference_run=reference,fit_root=fitted,device='cpu')
    assert revealed==order and trained==[21,22]
    done=real_read(out/'completed.json')
    assert done['source_and_decoder_unchanged'] and done['slow_frozen_parts_unchanged']
    assert done['eos_updates']==2 and done['label_shots']==22
    with np.load(out/'shots'/'0021_21.npz') as a,np.load(reference/'shots'/'0021_21.npz') as b:
        np.testing.assert_array_equal(a['inter'],b['static'])
        np.testing.assert_array_equal(a['both'],b['recorded_sampling_real'])


def test_risk_replay_includes_scored_startup_points_without_a_future_target_requirement():
    from run_risk_feature_joint_20261108 import ended_risk_records

    s=dict(signal(164),shot=1,device='EAST',times=np.arange(164)*.001)
    safe=ended_risk_records(s,(False,None))
    disruptive=ended_risk_records(s,(True,.163))
    assert [r['query'] for r in safe]==list(range(0,164,8))
    assert all(r['group']=='safe' and r['y']==0. for r in safe)
    assert disruptive[0]['query']==0 and disruptive[0]['group']=='earlier'
    assert any(r['query']<135 and r['group']=='positive' for r in disruptive)
    assert all((.163-s['times'][r['query']])*1000>30 for r in disruptive if r['y'])
    assert not any(r['query']>=136 for r in disruptive)
