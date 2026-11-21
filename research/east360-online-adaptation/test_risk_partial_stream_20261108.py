"""Full-stream integration contract for partial arrived-signal feedback."""

from copy import deepcopy

import numpy as np
import torch

from test_risk_meta_step_20261107 import teacher_case
from test_joint_stream_20261107 import signal


def test_partial_stream_uses_same_reference_before_any_outcome_access(tmp_path,monkeypatch):
    import run_risk_partial_stream_20261108 as run
    import run_joint_trajectory_20261107 as j

    model,_=teacher_case();base=model.risk.eval().requires_grad_(False)
    order=list(range(1,23));fitted=tmp_path/'fit';fitted.mkdir()
    torch.save(model.decoder.state_dict(),fitted/'prepared_recorded_sampling.pt')
    reference=tmp_path/'reference';reference.mkdir();(reference/'shots').mkdir()
    inputs={sid:dict(signal(190),shot=sid,times=np.arange(190)*.001) for sid in order}
    calls=[]

    def partial(base,decoder,s,*,warm=False):
        calls.append((s.get('shot',21),len(s['values']),warm))
        n=len(s['values']);curves={k:np.full(n,.05,np.float32) for k in run.ARMS}
        if not warm:
            for key,first in (('full',153),('partial_delayed',153),('partial_early',144)):
                curves[key][first:]=.8
        return curves,dict(updates={key:int(not warm) for key in run.ARMS[1:]})

    for rank,sid in enumerate(order,1):
        traces,_=partial(base,model.decoder,inputs[sid],warm=rank<=20)
        j.r.save_arrays(reference/'shots'/f'{rank:04d}_{sid}.npz',static=traces['static'],
                        recorded_sampling_real=traces['full'])
    calls.clear()
    j.r.write_new(reference/'protocol.json',dict(order=order,fit=str(fitted),warm=20,
        parent_sha256=j.PARENT_SHA256,threshold=j.r.THRESHOLD,deterministic_algorithms=True,
        source_files={str(j.r.ROOT/name):(j.r.ROOT/name).read_text() for name in
            ('run_risk_feature_intra_20261107.py','risk_feature_intra_20261107.py')}))
    j.r.write_new(reference/'completed.json',dict(status='SEALED_RISK_TEACHER360',
        records=[dict(shot=sid) for sid in order]))
    cfg=dict(order=order,raw_sha256={});archive=dict(sealed=[dict(shot=sid) for sid in order])
    monkeypatch.setattr(run,'partial_curves',partial)
    monkeypatch.setattr(j.r,'configure',lambda:torch.set_num_threads(1))
    monkeypatch.setattr(j.r,'load_parent',lambda _: (deepcopy(base),{'threshold':j.r.THRESHOLD}))
    monkeypatch.setattr(j,'cohort',lambda _: (reference,cfg,archive,('unused','unused',20)))
    monkeypatch.setattr(j,'raw_for',lambda sid,_:inputs[sid])
    monkeypatch.setattr(j.r,'prepare_actual',lambda raw,rank,mapping:dict(raw))
    real_read=j.r.read
    monkeypatch.setattr(j.r,'read',lambda p:dict(calibration_ids=order[:20])
                        if p==j.r.COLD/'input_map.json' else real_read(p))
    outcome_reads=[]
    monkeypatch.setattr(j.r.io23,'read_outcome',lambda *args:outcome_reads.append(args))
    out=tmp_path/'partial';run.predict(out,reference_run=reference,fit_root=fitted,device='cpu')
    assert not outcome_reads
    assert calls[:20]==[(sid,190,True) for sid in order[:20]]
    assert (21,154,False) in calls
    done=real_read(out/'completed.json')
    assert done['templates_unchanged'] and done['maximum_reference_error']==0.
    assert [v['shot'] for v in done['records']]==order
    assert done['witnesses'][0]['max_error']==0.

    # Only the scorer sees outcomes, after every curve has been sealed.
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
    assert result['metrics']['partial_early']['tp']==1
    assert result['metrics']['full']['tp']==0
    assert result['comparisons']['partial_early_vs_partial_delayed']['tp_gained']==[21]
