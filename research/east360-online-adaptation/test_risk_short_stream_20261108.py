"""Cross source-task construction and EAST startup clock without labels."""

from copy import deepcopy

import numpy as np
import torch

from test_joint_stream_20261107 import signal
from test_risk_meta_step_20261107 import teacher_case


def test_four_conditions_keep_training_and_inference_clock_separate():
    from run_risk_short_context_20261108 import comparison_curves
    from run_risk_feature_intra_20261107 import feature_curves

    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    model,_=teacher_case();base=model.risk.eval().requires_grad_(False)
    decoders={k:deepcopy(model.decoder).eval().requires_grad_(False) for k in ('control','short')}
    with torch.no_grad():
        decoders['short'].trajectory.bias.add_(.02)
    original=deepcopy(base.state_dict())
    curves,info=comparison_curves(base,decoders,signal(190))
    assert set(curves)=={'static','control_full','control_early','short_full','short_early'}
    for name,decoder in decoders.items():
        reference,_=feature_curves(base,decoder,signal(190),feedbacks=('real',))
        np.testing.assert_array_equal(curves[name+'_full'],reference['real'])
        assert info[name+'_full']['releases']==[153,185]
        assert info[name+'_early']['releases']==[57,89,121,153,185]
        np.testing.assert_allclose(curves[name+'_early'][:57],curves['static'][:57],atol=3e-6,rtol=0)
    assert all(torch.equal(v,original[k]) for k,v in base.state_dict().items())


def test_full_stream_seals_before_scoring_and_reproduces_its_control(tmp_path,monkeypatch):
    import run_risk_short_context_20261108 as run
    import run_joint_trajectory_20261107 as j
    from run_risk_feature_intra_20261107 import feature_curves
    from torch.nn.attention import sdpa_kernel,SDPBackend

    def cpu_configuration():
        torch.set_num_threads(1);torch.manual_seed(131)
        torch.backends.mha.set_fastpath_enabled(False)

    monkeypatch.setattr(j.r,'configure',cpu_configuration)
    cpu_configuration();torch.use_deterministic_algorithms(True)
    model,_=teacher_case();base=model.risk.eval().requires_grad_(False)
    control=model.decoder.eval().requires_grad_(False);short=deepcopy(control)
    with torch.no_grad():
        short.trajectory.bias.add_(.02)
    order=list(range(1,23));inputs={sid:dict(signal(190),shot=sid,times=np.arange(190)*.001) for sid in order}
    fitted=tmp_path/'fit';fitted.mkdir();reference_fit=tmp_path/'old_fit';reference_fit.mkdir()
    torch.save(control.state_dict(),reference_fit/'prepared_recorded_sampling.pt')
    for name,decoder in (('control',control),('short_context',short)):
        torch.save(decoder.state_dict(),fitted/f'prepared_{name}.pt')
    j.r.write_new(fitted/'protocol.json',dict(parent_sha256=j.PARENT_SHA256,
        reference_fit=str(reference_fit),cold20=order[:20],deterministic_algorithms=True,
        risk_classifier_training=False,east_label_fitting=False,jtext_online=False))
    j.r.write_new(fitted/'fit.json',dict(status='SOURCE_SHORT_CONTEXT_FIT_COMPLETE',
        source_control_exact=True,prepared_control_exact=True,source_classifier_unchanged=True))
    reference=tmp_path/'reference';reference.mkdir();(reference/'shots').mkdir()
    j.r.write_new(reference/'protocol.json',dict(order=order,fit=str(reference_fit),warm=20,
        parent_sha256=j.PARENT_SHA256,threshold=j.r.THRESHOLD,deterministic_algorithms=True,
        source_files={str(j.r.ROOT/name):(j.r.ROOT/name).read_text() for name in
            ('run_risk_feature_intra_20261107.py','risk_feature_intra_20261107.py')}))
    j.r.write_new(reference/'completed.json',dict(status='SEALED_RISK_TEACHER360',
        records=[dict(shot=sid) for sid in order]))
    with sdpa_kernel(SDPBackend.MATH):
        for rank,sid in enumerate(order,1):
            curves,_=feature_curves(base,control,inputs[sid],warm=rank<=20,feedbacks=('real',))
            j.r.save_arrays(reference/'shots'/f'{rank:04d}_{sid}.npz',static=curves['static'],
                            recorded_sampling_real=curves['real'])
    cfg=dict(order=order,raw_sha256={});archive=dict(sealed=[dict(shot=sid) for sid in order])
    monkeypatch.setattr(j.r,'load_parent',lambda _: (deepcopy(base),{'threshold':j.r.THRESHOLD}))
    monkeypatch.setattr(j,'cohort',lambda _: (reference,cfg,archive,('unused','unused',20)))
    monkeypatch.setattr(j,'raw_for',lambda sid,_:inputs[sid])
    monkeypatch.setattr(j.r,'prepare_actual',lambda raw,rank,mapping:dict(raw))
    real_read=j.r.read
    monkeypatch.setattr(j.r,'read',lambda p:dict(calibration_ids=order[:20])
                        if p==j.r.COLD/'input_map.json' else real_read(p))
    outcome_reads=[]
    monkeypatch.setattr(j.r.io23,'read_outcome',lambda *args:outcome_reads.append(args))
    out=tmp_path/'stream';run.predict(out,fit_root=fitted,reference_run=reference,device='cpu')
    assert not outcome_reads
    done=real_read(out/'completed.json')
    assert done['templates_unchanged'] and done['maximum_reference_error']<=5e-6
    assert [v['shot'] for v in done['records']]==order
    for row in done['records']:
        for key,value in row['adaptation'].items():
            assert value['updates']['real']==(0 if row['rank']<=20 else 5 if key.endswith('_early') else 2)
    assert max(w['max_error'] for w in done['witnesses'])<=5e-6

    truth={sid:dict(shot=sid,disruptive=sid%2==1,down=.180 if sid%2 else None) for sid in order}
    parent_rows={key:[] for key in ('static','recorded_sampling_real')}
    for rank,sid in enumerate(order,1):
        with np.load(reference/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as arrays:
            for key in parent_rows:
                parent_rows[key].append(j.score(inputs[sid]['times'],arrays[key],truth[sid],j.r.THRESHOLD))
    j.r.write_new(reference/'result.json',dict(rows=parent_rows))
    monkeypatch.setattr(j,'recorded',lambda ref,note,rank,real,stale:(inputs[note['shot']]['times'],None,None,None))
    monkeypatch.setattr(j.r.io23,'path_for',lambda device,sid:sid)

    def raw_truth(path,sid):
        assert len(list((out/'shots').glob('*.npz')))==len(order)
        outcome_reads.append(sid)
        return truth[sid],inputs[sid]['times']

    monkeypatch.setattr(j,'raw_truth',raw_truth)
    run.recount(out)
    assert outcome_reads==order
    result=real_read(out/'result.json')
    assert set(result['metrics'])==set(run.ARMS) and result['inter_updates']==0
