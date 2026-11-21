"""Synthetic, in-memory CPU checks for the independent anchored-stream audit."""

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

import ANCHORED_FULLSTREAM_AUDIT_20261031 as audit


VARIANTS = ('origin', 'mature')
ARMS = ('real', 'reverse', 'mean_feedback', 'donor')
NAMES = ('cold', 'identical', *(v + '__' + a for v in VARIANTS for a in ARMS))
SECTIONS = ('full360', 'cold20', 'suffix340', 'last90')


def test_strict_first_crossing_and_no_upper_limit_on_early_alarm():
    threshold = .5
    exact = audit.score_shot(1, np.array([.970, .975]), np.array([.5, .9]),
                             True, 1., threshold)
    timely = audit.score_shot(2, np.array([.969]), np.array([.5]), True, 1., threshold)
    early = audit.score_shot(3, np.array([5., 9.969]), np.array([.7, .9]),
                             True, 10., threshold)
    assert exact['alarm'] and not exact['timely'] and exact['lead_ms'] == 30.
    assert timely['timely'] and timely['lead_ms'] == 31.
    assert early['timely'] and early['lead_ms'] == 5000.
    assert early['first_alarm_time_seconds'] == 5.
    metrics = audit.summarize([exact, timely, early], threshold)
    assert (metrics['tp'], metrics['fn'], metrics['tardy'], metrics['early_over3000']) == (2, 1, 1, 1)
    assert metrics['median_lead_ms'] == 2515.5


def test_postterminal_crossing_is_not_an_alarm():
    row = audit.score_shot(1, np.array([.9, 1.001]), np.array([.1, .9]), True, 1., .5)
    assert not row['alarm'] and row['first_alarm_time_seconds'] is None


def test_lead_rounding_matches_saved_numpy_time_precision():
    row = audit.score_shot(1, np.array([0.], dtype=np.float64), np.array([.9]),
                           True, .0698144146865, .5)
    assert row['lead_ms'] == 69.814414686


def test_mean_control_does_not_duplicate_rescued_shots():
    cold = [dict(shot=1, is_disrupt=True, alarm=False, timely=False),
            dict(shot=2, is_disrupt=False, alarm=True, timely=False)]
    real = [dict(shot=1, is_disrupt=True, alarm=True, timely=True),
            dict(shot=2, is_disrupt=False, alarm=False, timely=False)]
    assert audit.paired_changes(cold, real) == dict(rescued=[1, 2], harmed=[])
    assert audit.paired_changes(cold, copy.deepcopy(cold)) == dict(rescued=[], harmed=[])
    assert audit.paired_changes(cold, real) == audit.paired_changes(copy.deepcopy(cold), real)


def test_actual_down_resolves_no_alarm_with_a_postterminal_crossing():
    times = np.array([0., 1.])
    no_alarm = dict(shot=7, is_disrupt=True, alarm=False, lead_ms=None,
                    timely=False, first_alarm_time_seconds=None)
    saved = {name: dict(no_alarm) for name in NAMES}
    curves = {name: np.array([.1, .9]) for name in NAMES}
    records = audit.check_shot_records(7, times, curves, saved,
                                       dict(shot=7, is_disrupt=True, down=.5), .5)
    assert all(not row['alarm'] for row in records.values())


def test_stale_consistent_saved_leads_rejected_against_actual_down():
    times = np.array([.9, .969, .970, 1.001])
    curves = {name: np.array([.1, .5, .6, .9]) for name in NAMES}
    expected = audit.score_shot(7, times, curves['cold'], True, 1., .5)
    saved = {name: dict(expected) for name in NAMES}
    outcome = dict(shot=7, is_disrupt=True, down=1.)
    records = audit.check_shot_records(7, times, curves, saved, outcome, .5)
    assert records['cold']['lead_ms'] == 31.
    assert records['cold']['first_alarm_time_seconds'] == .969
    for row in saved.values():
        row['lead_ms'] = 32.  # all arms agree, but all imply the wrong disruption time
    with pytest.raises(audit.AuditError, match='lead_ms'):
        audit.check_shot_records(7, times, curves, saved, outcome, .5)


def test_cli_requires_explicit_outcomes_without_opening_a_run(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['ANCHORED_FULLSTREAM_AUDIT_20261031.py'])
    def must_not_run(*args, **kwargs):
        raise AssertionError('argument parsing must not open any run')
    monkeypatch.setattr(audit, 'audit', must_not_run)
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 2
    assert '--outcomes' in capsys.readouterr().err


def update(event, supported=True):
    norms = {name: 0. for name in NAMES}
    cosines = {name: None for name in NAMES}
    if supported:
        for variant in VARIANTS:
            for arm in ('real', 'reverse', 'donor'):
                norms[variant + '__' + arm] = .05
                cosines[variant + '__' + arm] = 1.
    return dict(origin=event - 25, maturity=event, first_affected=event + 1,
                apply_count=min(25, 180 - event - 1), state_norm=norms, cos_real=cosines,
                max_bound=max(norms.values()), feedback_supported=supported,
                donor_index={'origin': 0, 'mature': 2})


def test_receipts_prove_clock_zero_replacement_and_norm_matching():
    curves = {name: np.full(180, .1, dtype=np.float32) for name in NAMES}
    for variant in VARIANTS:
        for arm in ('real', 'reverse', 'donor'):
            curves[variant + '__' + arm][153:178] = .2
    updates = [update(152), update(177, False)]
    counts = audit.validate_updates(updates, curves, startup=False)
    assert counts['origin__real'] == 1
    bad = copy.deepcopy(updates)
    bad[0]['state_norm']['origin__reverse'] = .02
    with pytest.raises(audit.AuditError, match='norm'):
        audit.validate_updates(bad, curves, startup=False)
    bad = copy.deepcopy(updates)
    bad[0]['first_affected'] = 152
    with pytest.raises(audit.AuditError, match='clock'):
        audit.validate_updates(bad, curves, startup=False)
    bad_curves = {name: value.copy() for name, value in curves.items()}
    bad_curves['origin__real'][178] = .2
    with pytest.raises(audit.AuditError, match='zero'):
        audit.validate_updates(updates, bad_curves, startup=False)


def test_norm_matching_is_relative_even_for_small_states():
    curves = {name: np.full(180, .1, dtype=np.float32) for name in NAMES}
    bad = update(152)
    bad['state_norm']['origin__real'] = 1e-8
    bad['state_norm']['origin__reverse'] = 9e-8
    bad['state_norm']['origin__donor'] = 0.
    bad['cos_real']['origin__donor'] = None
    with pytest.raises(audit.AuditError, match='norm'):
        audit.validate_updates([bad, update(177, False)], curves, startup=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class MemoryStore:
    def __init__(self):
        self.payloads = {}
        self.reads = []

    def put(self, path, content):
        self.payloads[Path(path)] = content

    def put_json(self, path, value):
        self.put(path, json.dumps(value, allow_nan=False).encode())

    def exists(self, path):
        return Path(path) in self.payloads

    def read_bytes(self, path):
        self.reads.append(Path(path))
        return self.payloads[Path(path)]

    def sha256(self, path):
        return digest(self.read_bytes(path))

    def files(self, directory):
        directory = Path(directory)
        return {path.name for path in self.payloads if path.parent == directory}


def npz(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def fixture_store(monkeypatch, *, with_rescue=False):
    fs, root = MemoryStore(), Path('/synthetic-worktree')
    output = root / 'probe_outputs/anchored_full360_diagnostic_v1'
    archive = root / 'probe_outputs/arrival_reference_east360_v1'
    reference_name = 'experiments/risk_aligned_online/runs/parent'
    monkeypatch.setattr(audit, 'REFERENCE_PATH', reference_name, raising=False)
    reference = root / reference_name
    ids = list(range(1001, 1361))
    # Exactly 17 disruptive and 3 safe in cold20; 180 of each overall.
    disruptive = set(ids[:17] + ids[20:183])
    threshold = audit.THRESHOLD
    source = root / 'experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt'
    fs.put(source, b'qualified source checkpoint')
    monkeypatch.setattr(audit, 'SOURCE_SHA256', fs.sha256(source))
    ref_payloads = {
        'protocol.json': b'{}', 'completed.json': b'{}',
        'order.json': json.dumps([{'shot': sid} for sid in ids]).encode(),
        'input_map.json': json.dumps({'calibration_ids': ids[:20]}).encode(),
        'input_hashes.json': b'{}', 'models/cold.pt': b'frozen slow state',
    }
    for name, data in ref_payloads.items():
        fs.put(reference / name, data)
    reference_hashes = {name: digest(data) for name, data in ref_payloads.items()}
    fs.put_json(archive / 'complete.json', {'complete': True})
    monkeypatch.setattr(audit, 'ARCHIVE_SHA256', fs.sha256(archive / 'complete.json'))
    fs.put_json(archive / 'protocol.json', dict(order=ids, reference=reference_name,
                                                reference_hashes=reference_hashes,
                                                threshold=threshold))
    monkeypatch.setattr(audit, 'ARCHIVE_PROTOCOL_SHA256', fs.sha256(archive / 'protocol.json'),
                        raising=False)
    code = b'fixed source\n'
    fs.put(root / 'frozen.py', code)
    snapshot = io.BytesIO()
    with zipfile.ZipFile(snapshot, 'w') as zipped:
        zipped.writestr('frozen.py', code)
    fs.put(output / 'code_snapshot.zip', snapshot.getvalue())
    protocol = dict(scope='SUPPLEMENTARY_FULL360_AFTER_FAILED_COMPONENT_SCREENS',
                    order=ids, curves=list(NAMES), variants=list(VARIANTS),
                    reference=reference_name, reference_hashes=reference_hashes,
                    archive=str(archive), archive_complete_sha256=audit.ARCHIVE_SHA256,
                    source_sha256=audit.SOURCE_SHA256, threshold=threshold, lead_ms=30.,
                    lead_comparison='strict greater than', warmup=20, source_updates=0,
                    inter_updates_after_cold20=0, alarm_updates=0,
                    independent_confirmation=False, component_screen='FAILED', promoted=False,
                    steps=256, batch=24, seed=131, lr=.001, eta=.1, radius=.1,
                    first_origin=127, horizon=25, first_affected='maturity+1',
                    reset_each_shot=True, replace_each_feedback=True,
                    all_predictions_before_evaluation_outcomes=True,
                    raw_startup_predictions_reused=True, code_sha256={'frozen.py': digest(code)})
    fs.put_json(output / 'protocol.json', protocol)
    times = np.arange(154 if with_rescue else 3, dtype=np.float64) / 1000.
    scores = np.full(len(times), .1, dtype=np.float32)
    original = npz(times=times, cold=scores)
    current = npz(times=times, **{name: scores for name in NAMES})
    archive_seals, seals, receipts = [], [], []
    rows = {name: [] for name in NAMES}
    for i, sid in enumerate(ids, 1):
        stem = f'{i:04d}_{sid}'
        old_file = archive / 'shots' / (stem + '.npz')
        old_seal = archive / 'shots' / (stem + '_seal.json')
        fs.put(old_file, original)
        fs.put_json(old_seal, dict(shot=sid, index=i, sha256=digest(original)))
        archive_seals.append(dict(shot=sid, sha256=digest(original)))
        updates = []
        shot_payload = current
        if with_rescue and i > 20:
            event = update(152, False)
            event['apply_count'] = 1
            if i == 21:
                event['feedback_supported'] = True
                event['state_norm']['origin__real'] = .05
                event['cos_real']['origin__real'] = 1.
                event['max_bound'] = .05
                changed = scores.copy()
                changed[153] = .9
                shot_payload = npz(times=times, **{name: changed if name == 'origin__real' else scores
                                                   for name in NAMES})
            updates = [event]
        fs.put(output / 'shots' / (stem + '.npz'), shot_payload)
        seal = dict(shot=sid, index=i, sha256=digest(shot_payload), updates=updates,
                    archive_seal_sha256=fs.sha256(old_seal), archive_max_abs_error=0.,
                    archive_crossing_disagreements=0, startup_archived=i <= 20)
        seal_path = output / 'shots' / (stem + '_seal.json')
        fs.put_json(seal_path, seal)
        seals.append(dict(shot=sid, sha256=digest(shot_payload)))
        receipts.append({k: v for k, v in seal.items() if k != 'updates'} |
                        {'seal_sha256': fs.sha256(seal_path)})
        for name in NAMES:
            rescued = with_rescue and i == 21 and name == 'origin__real'
            rows[name].append(dict(shot=sid, is_disrupt=sid in disruptive, alarm=rescued,
                                   lead_ms=847. if rescued else None, timely=rescued,
                                   first_alarm_time_seconds=.153 if rescued else None))
    fs.put_json(archive / 'all_predictions_sealed.json', dict(count=360, seals=archive_seals))
    monkeypatch.setattr(audit, 'ARCHIVE_BARRIER_SHA256',
                        fs.sha256(archive / 'all_predictions_sealed.json'), raising=False)
    fs.put_json(output / 'all_predictions_sealed.json', dict(count=360, seals=seals))
    fs.put_json(output / 'receipts.json', receipts)
    fs.put_json(output / 'accesses.json', [dict(shot=sid, sealed_count=360) for sid in ids])
    fs.put_json(output / 'per_shot.json', rows)
    for name in ('origin.pt', 'mature.pt', 'origin_training.json', 'mature_training.json'):
        fs.put(output / name, b'fixture artifact')

    def metrics(n_disrupt, n_safe):
        return dict(tp=0, fp=0, n_disrupt=n_disrupt, n_safe=n_safe,
                    tpr=0. if n_disrupt else None, fpr=0. if n_safe else None,
                    J=0. if n_disrupt and n_safe else None, threshold=threshold,
                    median_lead_ms=None, early_over3000=0)

    section_metrics = dict(full360=metrics(180, 180), cold20=metrics(17, 3),
                           suffix340=metrics(163, 177), last90=metrics(0, 90))
    empty = dict(rescued=[], harmed=[])
    comparisons = {section: {name: empty for name in NAMES} for section in SECTIONS}
    real_controls = {v: {a: {s: empty for s in SECTIONS} for a in ARMS[1:]} for v in VARIANTS}
    status = {v: dict(operational_improvement=False, real_exceeds_controls=False,
                      original_component_screen='FAILED', independent_confirmation=False)
              for v in VARIANTS}
    results = {name: copy.deepcopy(section_metrics) for name in NAMES}
    counts = {name: 0 for name in NAMES}
    if with_rescue:
        rescued = dict(rescued=[ids[20]], harmed=[])
        for section, disruptions in (('full360', 180), ('suffix340', 163)):
            result = results['origin__real'][section]
            result.update(tp=1, tpr=1 / disruptions, J=1 / disruptions, median_lead_ms=847.)
            comparisons[section]['origin__real'] = rescued
            for arm in ARMS[1:]:
                real_controls['origin'][arm][section] = rescued
        status['origin']['operational_improvement'] = True
        status['origin']['real_exceeds_controls'] = True
        counts['origin__real'] = 1
    complete = dict(complete=True, results=results,
                    comparisons=comparisons, real_vs_controls=real_controls,
                    operational_status=status, applied_updates=counts,
                    source_unchanged=True, component_screen='FAILED', promoted=False,
                    scientific_goal_achieved=False, independent_confirmation=False,
                    fpr_caps={name: {'0.1': True, '0.15': True, '0.2': True} for name in NAMES},
                    archive_max_abs_error=0.)
    complete['artifact_sha256'] = {name: fs.sha256(output / name) for name in fs.files(output)}
    fs.put_json(output / 'complete.json', complete)
    fs.put_json(root / 'authoritative_outcomes.json', dict(outcomes=[
        dict(shot=sid, is_disrupt=sid in disruptive, down=1. if sid in disruptive else None)
        for sid in ids]))
    fs.reads.clear()
    return fs, root, output, archive, reference


def test_full_360_in_memory_audit_and_seal_corruption(monkeypatch):
    fs, root, output, archive, reference = fixture_store(monkeypatch)
    report = audit.audit(output, outcomes=root / 'authoritative_outcomes.json',
                         root=root, store=fs, include_rows=True)
    assert report['verified'] is True
    assert report['checks']['shots'] == report['checks']['archived_shots'] == 360
    assert report['metrics']['cold']['full360']['n_disrupt'] == 180
    assert report['metrics']['cold']['full360']['n_safe'] == 180
    assert report['metrics']['cold']['cold20']['n_disrupt'] == 17
    assert report['metrics']['cold']['suffix340']['n_safe'] == 177
    assert report['comparisons']['full360']['origin__mean_feedback'] == dict(rescued=[], harmed=[])
    assert report['real_vs_controls']['origin']['mean_feedback']['full360'] == dict(rescued=[], harmed=[])
    assert len(report['per_shot']['cold']) == 360
    assert fs.reads.index(output / 'complete.json') < fs.reads.index(output / 'per_shot.json')
    assert fs.reads.index(output / 'complete.json') < fs.reads.index(root / 'authoritative_outcomes.json')
    assert report['checks']['outcomes_sha256'] == digest(fs.payloads[root / 'authoritative_outcomes.json'])

    shot = output / 'shots/0001_1001.npz'
    fs.put(shot, fs.payloads[shot] + b'corrupt')
    with pytest.raises(audit.AuditError, match='hash|digest|seal'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


def test_full_360_nonzero_rescue_and_real_versus_each_control(monkeypatch):
    fs, root, output, archive, reference = fixture_store(monkeypatch, with_rescue=True)
    report = audit.audit(output, outcomes=root / 'authoritative_outcomes.json',
                         root=root, store=fs, include_rows=True)
    assert report['metrics']['origin__real']['full360']['tp'] == 1
    assert report['metrics']['origin__real']['cold20']['tp'] == 0
    assert report['metrics']['origin__real']['suffix340']['tp'] == 1
    assert report['metrics']['origin__real']['last90']['tp'] == 0
    expected = dict(rescued=[1021], harmed=[])
    assert report['comparisons']['full360']['origin__real'] == expected
    assert report['comparisons']['full360']['origin__mean_feedback'] == dict(rescued=[], harmed=[])
    for arm in ('reverse', 'mean_feedback', 'donor'):
        assert report['real_vs_controls']['origin'][arm]['full360'] == expected
    assert report['per_shot']['origin__real'][20]['lead_ms'] == 847.
    assert report['checks']['applied_updates']['origin__real'] == 1


def shared_runs_fixture(monkeypatch, *, destination='/synthetic-main'):
    fs, root, output, archive, reference = fixture_store(monkeypatch)
    worktree_runs = root / 'experiments/risk_aligned_online/runs'
    shared_runs = Path('/synthetic-main/experiments/risk_aligned_online/runs')
    actual_runs = Path(destination) / 'experiments/risk_aligned_online/runs'
    for path, data in list(fs.payloads.items()):
        if path.is_relative_to(worktree_runs):
            del fs.payloads[path]
            fs.put(actual_runs / path.relative_to(worktree_runs), data)
    original_resolve = Path.resolve

    def resolve_with_runs_link(path, *args, **kwargs):
        resolved = original_resolve(path, *args, **kwargs)
        if resolved.is_relative_to(worktree_runs):
            return actual_runs / resolved.relative_to(worktree_runs)
        return resolved

    monkeypatch.setattr(Path, 'resolve', resolve_with_runs_link)
    monkeypatch.setattr(audit, 'SHARED_RUNS_ROOT', shared_runs, raising=False)
    return fs, root, output, shared_runs


def test_approved_runs_symlink_keeps_full_360_audit_and_hash_checks(monkeypatch):
    fs, root, output, shared = shared_runs_fixture(monkeypatch)
    report = audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)
    assert report['verified'] and report['checks']['reference_files'] == 6
    source = shared / 'gradient_aligned512_20260929/source.pt'
    fs.put(source, b'changed source')
    with pytest.raises(audit.AuditError, match='source hash'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


@pytest.mark.parametrize('name', ['completed.json', 'input_hashes.json', 'input_map.json',
                                  'models/cold.pt', 'order.json', 'protocol.json'])
def test_approved_runs_symlink_still_verifies_reference_hashes(monkeypatch, name):
    fs, root, output, shared = shared_runs_fixture(monkeypatch)
    fs.put(shared / 'parent' / name, b'changed reference')
    with pytest.raises(audit.AuditError, match='reference hash'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


def test_unapproved_runs_symlink_is_not_a_general_path_escape(monkeypatch):
    fs, root, output, shared = shared_runs_fixture(monkeypatch, destination='/untrusted')
    with pytest.raises(audit.AuditError, match='runs|escape'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


@pytest.mark.parametrize('name', [
    'experiments/risk_aligned_online/runs/other/source.pt',
    'experiments/risk_aligned_online/runs/other/data.hdf5',
    '../experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt',
    '/tmp/source.pt',
])
def test_shared_runs_exception_only_accepts_two_contracted_paths(name):
    with pytest.raises(audit.AuditError):
        audit.contracted_artifact(Path('/synthetic-worktree'), name)


@pytest.mark.parametrize('destination', [
    '/untrusted/source.pt',
    '/synthetic-main/experiments/risk_aligned_online/runs/unsafe.hdf5',
])
def test_nested_shared_artifact_symlink_cannot_escape_or_reach_hdf5(monkeypatch, destination):
    root = Path('/synthetic-worktree')
    worktree_runs = root / 'experiments/risk_aligned_online/runs'
    shared_runs = Path('/synthetic-main/experiments/risk_aligned_online/runs')
    source_relative = Path(audit.SOURCE_PATH).relative_to('experiments/risk_aligned_online/runs')
    original_resolve = Path.resolve

    def nested_symlink(path, *args, **kwargs):
        resolved = original_resolve(path, *args, **kwargs)
        if resolved.is_relative_to(worktree_runs):
            resolved = shared_runs / resolved.relative_to(worktree_runs)
        if resolved == shared_runs / source_relative:
            return Path(destination)
        return resolved

    monkeypatch.setattr(Path, 'resolve', nested_symlink)
    monkeypatch.setattr(audit, 'SHARED_RUNS_ROOT', shared_runs)
    with pytest.raises(audit.AuditError, match='escape|runs'):
        audit.contracted_artifact(root, audit.SOURCE_PATH)


def test_full_audit_rejects_stale_saved_lead_against_authoritative_file(monkeypatch):
    fs, root, output, archive, reference = fixture_store(monkeypatch, with_rescue=True)
    path = output / 'per_shot.json'
    saved = json.loads(fs.payloads[path])
    saved['origin__real'][20]['lead_ms'] = 848.
    fs.put_json(path, saved)
    complete = json.loads(fs.payloads[output / 'complete.json'])
    complete['artifact_sha256'][path.name] = fs.sha256(path)
    fs.put_json(output / 'complete.json', complete)
    with pytest.raises(audit.AuditError, match='1021/origin__real/lead_ms'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


def test_explicit_outcomes_sidecar_can_be_outside_the_sealed_manifest(monkeypatch):
    fs, root, output, archive, reference = fixture_store(monkeypatch)
    sidecar = output / 'authoritative_outcomes.json'
    fs.put(sidecar, fs.payloads[root / 'authoritative_outcomes.json'])
    report = audit.audit(output, outcomes=sidecar, root=root, store=fs)
    assert report['checks']['outcomes_sha256'] == digest(fs.payloads[sidecar])


def test_authoritative_outcomes_must_match_the_locked_cohort():
    ids = list(range(1, 361))
    rows = [dict(shot=sid, is_disrupt=sid <= 180, down=1. if sid <= 180 else None)
            for sid in ids]
    assert len(audit.validate_outcomes({'outcomes': rows}, ids)) == 360
    with_metadata = [{**rows[0], 'raw_sha256': 'metadata'}, *rows[1:]]
    assert len(audit.validate_outcomes({'outcomes': with_metadata}, ids)) == 360
    for altered in (rows[:-1], rows[:-1] + [rows[0]],
                    [{**rows[0], 'down': None}, *rows[1:]],
                    [*rows[:180], {**rows[180], 'down': 1.}, *rows[181:]]):
        with pytest.raises(audit.AuditError):
            audit.validate_outcomes({'outcomes': altered}, ids)


def test_saved_disruption_flags_are_checked_against_authoritative_outcomes(monkeypatch):
    fs, root, output, archive, reference = fixture_store(monkeypatch)
    path = root / 'authoritative_outcomes.json'
    payload = json.loads(fs.payloads[path])
    payload['outcomes'][0].update(is_disrupt=False, down=None)
    payload['outcomes'][-1].update(is_disrupt=True, down=1.)
    fs.put_json(path, payload)  # class totals stay 180/180 but identities differ
    with pytest.raises(audit.AuditError, match='per_shot/1001/cold/is_disrupt'):
        audit.audit(output, outcomes=path, root=root, store=fs)


def test_incomplete_output_never_reads_outcomes():
    fs = MemoryStore()
    root, output = Path('/root'), Path('/root/probe_outputs/incomplete')
    with pytest.raises(audit.AuditError, match='complete'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)
    assert not fs.reads


def test_reference_or_snapshot_hash_changes_fail(monkeypatch):
    fs, root, output, archive, reference = fixture_store(monkeypatch)
    fs.put(reference / 'models/cold.pt', b'changed model')
    with pytest.raises(audit.AuditError, match='hash|digest'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


def test_barrier_links_are_checked_even_if_manifest_is_rehashed(monkeypatch):
    fs, root, output, archive, reference = fixture_store(monkeypatch)
    path = output / 'all_predictions_sealed.json'
    barrier = json.loads(fs.payloads[path])
    barrier['seals'][0]['sha256'] = '0' * 64
    fs.put_json(path, barrier)
    complete = json.loads(fs.payloads[output / 'complete.json'])
    complete['artifact_sha256'][path.name] = fs.sha256(path)
    fs.put_json(output / 'complete.json', complete)
    with pytest.raises(audit.AuditError, match='barrier'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


def test_protocol_threshold_must_be_exact_even_if_manifest_is_rehashed(monkeypatch):
    fs, root, output, archive, reference = fixture_store(monkeypatch)
    path = output / 'protocol.json'
    protocol = json.loads(fs.payloads[path])
    protocol['threshold'] += 1e-9
    fs.put_json(path, protocol)
    complete = json.loads(fs.payloads[output / 'complete.json'])
    complete['artifact_sha256'][path.name] = fs.sha256(path)
    fs.put_json(output / 'complete.json', complete)
    with pytest.raises(audit.AuditError, match='threshold'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


@pytest.mark.parametrize('name', ['protocol.json', 'all_predictions_sealed.json'])
def test_prior_archive_metadata_is_independently_hash_locked(monkeypatch, name):
    fs, root, output, archive, reference = fixture_store(monkeypatch)
    path = archive / name
    metadata = json.loads(fs.payloads[path])
    metadata['unrecorded_change'] = True
    fs.put_json(path, metadata)
    with pytest.raises(audit.AuditError, match='archive.*hash'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


@pytest.mark.parametrize('relative', ['per_shot.json', 'shots/0001_1001.npz'])
def test_bytes_used_for_rescoring_are_the_bytes_that_were_sealed(monkeypatch, relative):
    fs, root, output, archive, reference = fixture_store(monkeypatch)
    target, reads = output / relative, 0
    actual_read = fs.read_bytes

    def changing_read(path):
        nonlocal reads
        value = actual_read(path)
        if Path(path) == target:
            reads += 1
            if reads == 2:
                return value + b' '
        return value

    monkeypatch.setattr(fs, 'read_bytes', changing_read)
    with pytest.raises(audit.AuditError, match='hash'):
        audit.audit(output, outcomes=root / 'authoritative_outcomes.json', root=root, store=fs)


@pytest.mark.parametrize('name', ['../secret', '/absolute', 'file.hdf5', 'file.h5'])
def test_unsafe_or_hdf5_artifact_paths_are_rejected(name):
    with pytest.raises(audit.AuditError):
        audit.safe_child(Path('/synthetic'), name)
