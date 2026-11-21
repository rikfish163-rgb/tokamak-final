"""Re-teaching and changing the runtime direction are separate factors."""

from copy import deepcopy

import numpy as np
import torch

from test_joint_stream_20261107 import signal
from test_risk_meta_step_20261107 import teacher_case


def test_three_new_conditions_keep_common_clock_and_reference_baseline():
    from run_risk_direction_stream_20261108 import comparison_curves
    from run_risk_feature_intra_20261107 import feature_curves

    torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
    model, _ = teacher_case()
    base = model.risk.eval().requires_grad_(False)
    gs = dict(control=model.decoder.eval().requires_grad_(False), direction=deepcopy(model.decoder))
    with torch.no_grad():
        gs['direction'].trajectory.bias.add_(.03)
    s = signal(190)
    old, old_info = feature_curves(base, gs['control'], s, feedbacks=('real',))
    curves, info = comparison_curves(base, gs, s, reference=old)
    assert set(curves) == {'static', 'control_adam', 'control_direction', 'taught_adam', 'taught_direction'}
    np.testing.assert_array_equal(curves['control_adam'], old['real'])
    assert set(info) == {'control_direction', 'taught_adam', 'taught_direction'}
    for row in info.values():
        assert row['releases'] == old_info['releases'] == [153, 185]
        assert row['updates']['real'] == 2
    direct, _ = feature_curves(base, gs['direction'], s, feedbacks=('real',))
    np.testing.assert_array_equal(curves['taught_adam'], direct['real'])


def test_stream_seals_without_outcomes_and_scores_all_original_members(tmp_path, monkeypatch):
    import run_risk_direction_stream_20261108 as run
    import run_joint_trajectory_20261107 as j
    from run_risk_feature_intra_20261107 import feature_curves

    def cpu_config():
        torch.set_num_threads(1); torch.manual_seed(131)
        torch.backends.mha.set_fastpath_enabled(False)
    monkeypatch.setattr(j.r, 'configure', cpu_config)
    cpu_config()
    model, _ = teacher_case()
    base = model.risk.eval().requires_grad_(False)
    g = model.decoder.eval().requires_grad_(False)
    order = list(range(1, 23))
    inputs = {sid:dict(signal(190), shot=sid, times=np.arange(190)*.001) for sid in order}
    fit = tmp_path/'fit'; fit.mkdir()
    for name in ('control', 'direction'):
        torch.save(g.state_dict(), fit/f'prepared_{name}.pt')
    j.r.write_new(fit/'protocol.json', dict(parent_sha256=j.PARENT_SHA256, cold20=order[:20],
        source_steps=512, sequence_steps=4, risk_classifier_training=False, east_label_fitting=False,
        jtext_online=False, deterministic_algorithms=True))
    j.r.write_new(fit/'fit.json', dict(status='SOURCE_DIRECTION_TEACHER_FIT_COMPLETE',
        source_control_exact=True, prepared_control_exact=True, source_classifier_unchanged=True))
    ref = tmp_path/'ref'; ref.mkdir(); (ref/'shots').mkdir()
    j.r.write_new(ref/'protocol.json', dict(order=order, threshold=j.r.THRESHOLD, warm=20,
        parent_sha256=j.PARENT_SHA256, inter_updates=0, deterministic_algorithms=True,fit=str(fit),
        source_files={str(j.r.ROOT/name):(j.r.ROOT/name).read_text() for name in
                      ('run_risk_feature_intra_20261107.py','risk_feature_intra_20261107.py')}))
    records = []; old_rows = {name:[] for name in ('static','control_full')}
    truth = {sid:dict(shot=sid, disruptive=sid%2==1, down=.180 if sid%2 else None) for sid in order}
    for rank, sid in enumerate(order, 1):
        curves, info = feature_curves(base, g, inputs[sid], warm=rank<=20, feedbacks=('real',))
        j.r.save_arrays(ref/'shots'/f'{rank:04d}_{sid}.npz', static=curves['static'], control_full=curves['real'])
        records.append(dict(shot=sid, rank=rank, adaptation=dict(control_full=info)))
        for new, old in (('static','static'), ('control_full','real')):
            old_rows[new].append(j.score(inputs[sid]['times'], curves[old], truth[sid], j.r.THRESHOLD))
    j.r.write_new(ref/'completed.json', dict(status='SEALED_SOURCE_CONTEXT_CLOCK360', records=records, templates_unchanged=True))
    j.r.write_new(ref/'result.json', dict(rows=old_rows))
    cfg = dict(order=order, raw_sha256={})
    monkeypatch.setattr(j, 'cohort', lambda _: (ref, cfg, dict(sealed=[dict(shot=s) for s in order]), (None,None,20)))
    monkeypatch.setattr(j.r, 'load_parent', lambda _: (deepcopy(base), dict(threshold=j.r.THRESHOLD)))
    monkeypatch.setattr(j, 'raw_for', lambda sid,_: inputs[sid])
    monkeypatch.setattr(j.r, 'prepare_actual', lambda raw,rank,mapping: dict(raw))
    read = j.r.read
    monkeypatch.setattr(j.r,'read',lambda p:dict(calibration_ids=order[:20]) if p==j.r.COLD/'input_map.json' else read(p))
    out = tmp_path/'out'
    reads = []
    monkeypatch.setattr(j.r.io23,'read_outcome',lambda *args:reads.append(args))
    run.predict(out, fit_root=fit, reference_run=ref, device='cpu')
    assert not reads
    done = read(out/'completed.json')
    assert [v['shot'] for v in done['records']] == order and done['templates_unchanged']
    monkeypatch.setattr(j,'recorded',lambda ref,note,rank,real,stale:(inputs[note['shot']]['times'],None,None,None))
    monkeypatch.setattr(j.r.io23,'path_for',lambda device,sid:sid)
    def raw_truth(path, sid):
        assert len(list((out/'shots').glob('*.npz'))) == len(order)
        reads.append(sid)
        return truth[sid], inputs[sid]['times']
    monkeypatch.setattr(j,'raw_truth',raw_truth)
    run.recount(out)
    assert reads == order
    result = read(out/'result.json')
    assert set(result['metrics']) == set(run.ARMS) and result['inter_updates'] == 0
    assert len(result['rows']['taught_direction']) == len(order)
