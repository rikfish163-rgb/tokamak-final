"""Synthetic CPU checks; never open a live experiment or actual shot data."""

import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.risk_aligned_online.context_vector_20261030 import audit_full as audit


def test_first_alarm_strict_30ms_and_preterminal_mask():
    times = np.array([9.969, 9.970, 10.000, 10.001])
    at_30 = audit.shot_decision(1, times, [0, 0.5, 0, 1], True, 10.0, 0.5)
    assert at_30['lead_ms'] == 30.0
    assert at_30['alarm'] and not at_30['timely']
    earlier = audit.shot_decision(1, times, [0.5, 1, 0, 0], True, 10.0, 0.5)
    assert earlier['lead_ms'] == 31.0 and earlier['timely']
    after_only = audit.shot_decision(1, times, [0, 0, 0, 1], True, 10.0, 0.5)
    assert not after_only['alarm'] and after_only['lead_ms'] is None


def test_no_alarm_safe_false_positive_and_saved_counts():
    times = np.array([0.0, 0.1])
    rows = [audit.shot_decision(1, times, [0, 0], True, 1.0, 0.5),
            audit.shot_decision(2, times, [0, 0.5], False, None, 0.5),
            audit.shot_decision(3, times, [0, 0], False, None, 0.5)]
    metrics = audit.summarize(rows, 0.5)
    assert (metrics['tp'], metrics['fp'], metrics['n_disrupt'], metrics['n_safe']) == (0, 1, 1, 2)
    assert metrics['tpr'] == 0 and metrics['fpr'] == 0.5 and metrics['J'] == -0.5
    assert metrics['median_lead_ms'] is None
    audit.compare_metrics(metrics, copy.deepcopy(metrics))
    altered = copy.deepcopy(metrics)
    altered['fp'] = 0
    with pytest.raises(ValueError, match='fp'):
        audit.compare_metrics(metrics, altered)
    altered['fp'] = 1.0 + 1e-10
    with pytest.raises(ValueError, match='fp'):
        audit.compare_metrics(metrics, altered)


def _receipt(event, length, supported=False):
    return dict(origin=event - 25, maturity=event, first_affected=event + 1,
                apply_count=min(25, length - event - 1), feedback_supported=supported,
                state_norm={name: 0.0 for name in audit.NAMES},
                cos_real={name: None for name in audit.NAMES}, max_bound=0.0)


def _curves(length=180):
    cold = np.linspace(0.1, 0.4, length, dtype=np.float32)
    return np.arange(length) / 1000, {name: cold.copy() for name in audit.NAMES}


def test_causal_prefix_and_support_reset_are_checked_against_cold():
    times, curves = _curves()
    updates = [_receipt(152, len(times)), _receipt(177, len(times))]
    audit.verify_curve_contract(times, curves, times.copy(), curves['cold'].copy(), updates, False)
    changed = {name: values.copy() for name, values in curves.items()}
    changed['raw_fixed__real'][152] += 0.01
    with pytest.raises(ValueError, match='prefix'):
        audit.verify_curve_contract(times, changed, times, curves['cold'], updates, False)
    changed = {name: values.copy() for name, values in curves.items()}
    changed['raw_fixed__real'][153] += 0.01
    with pytest.raises(ValueError, match='unsupported|reset'):
        audit.verify_curve_contract(times, changed, times, curves['cold'], updates, False)


def test_receipt_cadence_and_first_affected_are_required():
    times, curves = _curves()
    updates = [_receipt(152, len(times)), _receipt(177, len(times))]
    updates[1]['first_affected'] = 177
    with pytest.raises(ValueError, match='receipt|cadence|timing'):
        audit.verify_curve_contract(times, curves, times, curves['cold'], updates, False)


def test_paired_rescues_and_harms_include_disruptions_and_safe_shots():
    times = np.array([0.0])
    old = [audit.shot_decision(1, times, [0], True, 1, .5),
           audit.shot_decision(2, times, [0], False, None, .5)]
    new = [audit.shot_decision(1, times, [1], True, 1, .5),
           audit.shot_decision(2, times, [1], False, None, .5)]
    assert audit.paired_changes(old, new) == {'rescued': [1], 'harmed': [2]}


def test_missing_completion_fails_before_other_files_are_read(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError('other files or outcomes must not be read')

    monkeypatch.setattr(audit, 'read_json', forbidden)
    with pytest.raises(FileNotFoundError, match='complete'):
        audit.require_completed_barrier(tmp_path)


def test_invalid_global_barrier_blocks_before_outcome_access(tmp_path, monkeypatch):
    ids = list(range(1, 361))
    protocol = tmp_path / 'protocol.json'
    barrier = tmp_path / 'all_predictions_sealed.json'
    protocol.write_text(json.dumps({'order': ids}))
    barrier.write_text(json.dumps({'count': 359, 'seals': [dict(shot=s, sha256='a' * 64) for s in ids],
                                   'receipt_sha256': ['b' * 64] * 360}))
    artifacts = {name: 'unused' for name in ('protocol.json', 'code_snapshot.zip',
                 'all_predictions_sealed.json', 'per_shot.json', 'receipts.json', 'accesses.json')}
    artifacts['protocol.json'] = audit.sha(protocol)
    artifacts['all_predictions_sealed.json'] = audit.sha(barrier)
    (tmp_path / 'complete.json').write_text(json.dumps({'complete': True, 'artifact_sha256': artifacts}))
    monkeypatch.setattr(audit.io23, 'read_outcome', lambda *_: pytest.fail('premature outcome read'))
    with pytest.raises(ValueError, match='global360'):
        audit.require_completed_barrier(tmp_path)


def test_cli_will_not_overwrite_a_prior_audit(tmp_path, monkeypatch):
    output = tmp_path / 'audit.json'
    output.write_text('preserved')
    monkeypatch.setattr(audit, 'RUN', tmp_path)
    monkeypatch.setattr('sys.argv', ['audit_full', '--output', str(output)])
    monkeypatch.setattr(audit, 'audit', lambda *_: pytest.fail('existing output must fail first'))
    with pytest.raises(FileExistsError):
        audit.main()
    assert output.read_text() == 'preserved'


def test_all_360_receipt_and_archive_identities_are_checked(tmp_path, monkeypatch):
    directory, previous = tmp_path / 'run', tmp_path / 'previous'
    ids = list(range(1, 361))
    new_digest, new_seal = 'a' * 64, 'b' * 64
    old_digest, old_seal = 'c' * 64, 'd' * 64
    barrier = {'seals': [dict(shot=s, sha256=new_digest) for s in ids],
               'receipt_sha256': [new_seal] * 360}
    old_barrier = {'seals': [dict(shot=s, sha256=old_digest) for s in ids]}
    seals = [dict(shot=s, index=i + 1, sha256=new_digest, updates=[],
                  archive_seal_sha256=old_seal, archive_max_abs_error=0.,
                  archive_crossing_disagreements=0, startup_archived=i < 20,
                  outcomes_used_in_prediction=False) for i, s in enumerate(ids)]
    receipts = [{**{k: v for k, v in seal.items() if k != 'updates'}, 'seal_sha256': new_seal}
                for seal in seals]
    monkeypatch.setattr(audit, 'PREVIOUS', previous)

    def digest(path):
        path = str(path)
        old = str(previous) in path
        return (old_seal if old else new_seal) if path.endswith('_seal.json') else (old_digest if old else new_digest)

    def read(path):
        path = str(path)
        if path.endswith('/receipts.json'):
            return receipts
        if path.endswith('/accesses.json'):
            return [dict(shot=s, sealed_count=360) for s in ids]
        sid = int(path.rsplit('_', 2)[-2])
        if str(previous) in path:
            return dict(shot=sid, index=sid, sha256=old_digest)
        return seals[sid - 1]

    monkeypatch.setattr(audit, 'sha', digest)
    monkeypatch.setattr(audit, 'read_json', read)
    assert audit.verify_all_seals(directory, {'order': ids}, barrier, old_barrier) == receipts
    receipts[210]['index'] = 999
    with pytest.raises(ValueError, match='receipt identity'):
        audit.verify_all_seals(directory, {'order': ids}, barrier, old_barrier)


def test_provenance_checks_preserve_failed_component_gates(tmp_path, monkeypatch):
    root, directory = tmp_path, tmp_path / 'run'
    previous, policy, parent, reference = (root / 'previous', root / 'policy', root / 'parent', root / 'ref')
    for name, value in [('ROOT', root), ('PREVIOUS', previous), ('POLICY', policy), ('PARENT', parent)]:
        monkeypatch.setattr(audit, name, value)
    ids = list(range(1, 361))
    plan_name, plan_data = 'CONTEXT_VECTOR_FULLSTREAM_DIAGNOSTIC_PLAN_20261030.md', b'plan'
    plan_hash = hashlib.sha256(plan_data).hexdigest()
    reference_hashes = {'order.json': 'ref', 'input_map.json': 'ref'}
    protocol = dict(scope='DIAGNOSTIC_FULL360_AFTER_FAILED_COMPONENT_GATE', curves=list(audit.NAMES),
                    variants=list(audit.VARIANTS), update_arms=list(audit.ACTIONS), order=ids,
                    component_gate_status={v: 'FAILED' for v in audit.VARIANTS}, promoted=False,
                    independent_confirmation=False, reset_each_shot=True, replace_each_feedback=True,
                    no_feedback_replaces_with_zero=True, startup_reused_from_sealed_archive=True,
                    outcomes_after_all_predictions=True, warmup=20, first_origin=127, horizon=25,
                    update_cadence=25, first_affected='maturity + 1', lead_ms=30.,
                    lead_comparison='strict greater than', source_updates=0, inter_updates_after_cold20=0,
                    alarm_updates=0, source_sha256=audit.PARENT_SHA, code_sha256={plan_name: plan_hash},
                    plan_sha256=plan_hash, previous=str(previous), previous_complete_sha256=audit.PREVIOUS_SHA,
                    previous_hashes={'protocol.json': 'old', 'all_predictions_sealed.json': 'old'},
                    reference=str(reference), reference_hashes=reference_hashes, threshold=.5,
                    policy=str(policy), policy_hashes={'complete.json': 'policy'})
    complete = dict(scope=protocol['scope'], component_gate_status=protocol['component_gate_status'],
                    promoted=False, scientific_goal_achieved=False, independent_confirmation=False,
                    source_unchanged=True, fixed_cold_unchanged=True, source_parameter_updates=0,
                    artifact_sha256={name: 'artifact' for name in ('code_snapshot.zip', 'receipts.json', 'accesses.json')})
    old_protocol = dict(order=ids, source_sha256=audit.PARENT_SHA, threshold=.5,
                        reference=str(reference), reference_hashes=reference_hashes)
    old_barrier = dict(count=360, seals=[{'shot': s} for s in ids])
    policy_complete = dict(complete=True, source_unchanged=True, full_stream_run=False,
                           suffix_labels_read=0, selected_for_full_stream=None,
                           variants={v: {'full_stream_gate': False, 'reasons': ['failed']}
                                     for v in audit.VARIANTS})

    def digest(path):
        path = str(path)
        if path == str(parent):
            return audit.PARENT_SHA
        if path == str(root / plan_name):
            return plan_hash
        if path == str(previous / 'complete.json'):
            return audit.PREVIOUS_SHA
        if path.startswith(str(previous) + '/'):
            return 'old'
        if path.startswith(str(reference) + '/'):
            return 'ref'
        if path.startswith(str(policy) + '/'):
            return 'policy'
        if path.startswith(str(directory) + '/'):
            return 'artifact'
        raise AssertionError(path)

    def read(path):
        path = str(path)
        return {str(previous / 'protocol.json'): old_protocol,
                str(previous / 'all_predictions_sealed.json'): old_barrier,
                str(reference / 'order.json'): [{'shot': s} for s in ids],
                str(reference / 'input_map.json'): {'calibration_ids': ids[:20]},
                str(policy / 'complete.json'): policy_complete}[path]

    class Snapshot:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def namelist(self):
            return [plan_name]

        def read(self, name):
            assert name == plan_name
            return plan_data

    monkeypatch.setattr(audit, 'sha', digest)
    monkeypatch.setattr(audit, 'read_json', read)
    monkeypatch.setattr(audit.zipfile, 'ZipFile', Snapshot)
    _, _, reasons = audit.verify_provenance(directory, complete, protocol)
    assert all(reasons[v] == ['failed'] for v in audit.VARIANTS)
    policy_complete['variants']['raw_fixed']['full_stream_gate'] = True
    with pytest.raises(ValueError, match='failure gates'):
        audit.verify_provenance(directory, complete, protocol)


def test_full360_recount_on_sealed_in_memory_synthetic_curves(tmp_path, monkeypatch):
    """Exercise all arms/segments without opening any real data or writing an audit."""
    directory, previous, reference = tmp_path / 'run', tmp_path / 'previous', tmp_path / 'reference'
    ids = list(range(1, 361))
    times = np.arange(154) / 1000
    cold = np.full(154, .1, dtype=np.float32)
    changed_arm = 'raw_fixed__real'
    records = {}
    for arm in audit.NAMES:
        records[arm] = []
        for sid in ids:
            alarm = arm == changed_arm and sid in (21, 181)
            records[arm].append(dict(shot=sid, is_disrupt=sid <= 180, alarm=alarm,
                                     lead_ms=847.0 if alarm and sid <= 180 else None,
                                     timely=alarm and sid <= 180,
                                     first_alarm_time_seconds=.153 if alarm else None))

    results, paired = {}, {part: {} for part in audit.SECTIONS}
    caps = {}
    for arm in audit.NAMES:
        results[arm], caps[arm] = {}, {}
        for part, cut in audit.SECTIONS.items():
            section = records[arm][cut]
            nd = sum(r['is_disrupt'] for r in section)
            ns = len(section) - nd
            tp = sum(r['timely'] for r in section)
            fp = sum(r['alarm'] and not r['is_disrupt'] for r in section)
            tpr, fpr = (tp / nd if nd else None), (fp / ns if ns else None)
            results[arm][part] = dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns, tpr=tpr, fpr=fpr,
                                      J=tpr - fpr if nd and ns else None, threshold=.5,
                                      median_lead_ms=847.0 if tp else None, early_over3000=0)
            caps[arm][part] = {str(cap): None if fpr is None else fpr <= cap for cap in (.1, .15, .2)}
            section_ids = {r['shot'] for r in section}
            paired[part][arm] = dict(rescued=[21] if arm == changed_arm and 21 in section_ids else [],
                                     harmed=[181] if arm == changed_arm and 181 in section_ids else [])
    controls = {v: {control: {part: copy.deepcopy(paired[part][v + '__real'])
                              for part in audit.SECTIONS}
                    for control in audit.ACTIONS[1:]} for v in audit.VARIANTS}
    counts = {name: 2 if name == changed_arm else 0 for name in audit.NAMES}
    protocol = {'order': ids, 'threshold': .5}
    complete = dict(artifact_sha256={'per_shot.json': 'per-shot', 'all_predictions_sealed.json': 'barrier'},
                    results=results, comparisons=paired, real_vs_controls=controls, fpr_cap_status=caps,
                    update_counts=counts, applied_update_counts=counts, applied_point_counts=counts,
                    max_state_norm=.1, archive_max_abs_error=0, archive_crossing_disagreements=0,
                    source_unchanged=True, fixed_cold_unchanged=True,
                    component_gate_status={v: 'FAILED' for v in audit.VARIANTS})
    barrier = {'seals': [dict(shot=s, sha256='new') for s in ids]}
    old_barrier = {'seals': [dict(shot=s, sha256='old') for s in ids]}
    receipts = [{'seal_sha256': 'seal', 'archive_seal_sha256': 'archive-seal'} for _ in ids]
    ready = {'sealed': False}
    monkeypatch.setattr(audit, 'PREVIOUS', previous)
    monkeypatch.setattr(audit, 'require_completed_barrier', lambda *_: (complete, protocol, barrier))
    monkeypatch.setattr(audit, 'verify_provenance',
                        lambda *_: (reference, old_barrier, {v: ['failed'] for v in audit.VARIANTS}))

    def seal_all(*_):
        ready['sealed'] = True
        return receipts

    monkeypatch.setattr(audit, 'verify_all_seals', seal_all)

    def read(path):
        path = str(path)
        if path.endswith('per_shot.json'):
            assert ready['sealed']
            return records
        if path == str(audit.io23.EAST_COHORT):
            assert ready['sealed']
            return {'test': {'disruption_shots': ids[:180], 'non_disruption_shots': ids[180:]}}
        if path.endswith('input_hashes.json'):
            return {str(s): 'raw' for s in ids}
        if path.endswith('_seal.json'):
            sid = int(path.rsplit('_', 2)[-2])
            if sid <= 20:
                return {'updates': []}
            item = _receipt(152, len(times), supported=sid in (21, 181))
            if item['feedback_supported']:
                item['state_norm'][changed_arm] = .1
                item['cos_real'][changed_arm] = 1.
                item['max_bound'] = .1
            return {'updates': [item]}
        raise AssertionError(path)

    def digest(path):
        path = str(path)
        if path == str(audit.PARENT):
            return audit.PARENT_SHA
        if path == str(audit.io23.EAST_COHORT):
            return audit.MANIFEST_SHA
        if path.endswith('complete.json'):
            return 'completion'
        if path.endswith('per_shot.json'):
            return 'per-shot'
        if path.endswith('_seal.json'):
            return 'seal'
        if path.endswith('.hdf5'):
            return 'raw'
        return 'parent'

    def saved_npz(path, _digest):
        path = str(path)
        sid = int(path.rsplit('_', 1)[-1].split('.')[0])
        if str(previous) in path:
            return {'times': times, 'cold': cold}
        curves = {arm: cold.copy() for arm in audit.NAMES}
        if sid in (21, 181):
            curves[changed_arm][153] = .5
        return {'times': times, **curves}

    def outcome(_device, sid):
        assert ready['sealed']
        return sid <= 180, 1.0 if sid <= 180 else None

    monkeypatch.setattr(audit, 'read_json', read)
    monkeypatch.setattr(audit, 'sha', digest)
    monkeypatch.setattr(audit, 'verified_npz', saved_npz)
    monkeypatch.setattr(audit.io23, 'path_for', lambda _d, sid: tmp_path / f'{sid}.hdf5')
    monkeypatch.setattr(audit.io23, 'read_outcome', outcome)
    report = audit.audit(directory)
    assert report['audit_passed'] and report['checks']['per_shot_decisions_compared'] == 7920
    assert len(report['seal_identities']) == 360
    assert report['comparisons']['full360'][changed_arm] == {'rescued': [21], 'harmed': [181]}
    assert report['confusion_counts'][changed_arm]['full360'] == {'tp': 1, 'fn': 179, 'fp': 1, 'tn': 179}
    assert report['real_vs_controls']['raw_fixed']['reverse']['suffix340'] == {
        'rescued': [21], 'harmed': [181]}
    assert not report['promoted'] and not report['independent_confirmation']
    complete['results'][changed_arm]['full360']['tp'] = 2
    with pytest.raises(ValueError, match='raw_fixed__real/full360: metric tp'):
        audit.audit(directory)
